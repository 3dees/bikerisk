"""
LLM-based requirement consolidation for harmonization layer.

This module takes groups of similar clauses (from grouping.py) and uses an LLM
to create consolidated "core requirements" that unify the regulatory intent
across multiple standards.
"""

import json
from typing import List, Callable, Optional, Dict, Any

from harmonization.models import Clause, RequirementGroup
from harmonization.grouping import CATEGORY_TITLE_MAP


def build_llm_prompt_for_group(
    clauses: List[Clause],
    group_index: int
) -> tuple[str, str]:
    """
    Build system + user prompts for consolidating cross-standard requirement groups.
    
    Returns a tuple of (system_prompt, user_prompt) for Anthropic API.
    Uses new differences_across_standards schema with standard_id, clause_labels, difference_summary.

    Args:
        clauses: List of similar Clause objects to consolidate
        group_index: Index of this group (for logging/tracking)

    Returns:
        Tuple of (system_prompt_string, user_prompt_string)
    """
    # Verify this is a cross-standard group
    standards = [cl.standard_name for cl in clauses]
    unique_standards = sorted(set(standards))

    if len(unique_standards) < 2:
        raise ValueError(
            f"Group {group_index} contains clauses from only {unique_standards}. "
            "Cross-standard groups require 2+ standards."
        )

    # Build clause text list
    numbered = []
    for idx, cl in enumerate(clauses, 1):
        line = f"{idx}. [{cl.standard_name}] {cl.clause_number}: {cl.text}"
        numbered.append(line)
    clause_block = "\n".join(numbered)

    # Extract category and requirement_type metadata
    category_label = "Unknown"
    requirement_type_label = "unknown"
    if clauses and clauses[0].category:
        category = clauses[0].category
        category_label = CATEGORY_TITLE_MAP.get(category, category)
    if clauses and clauses[0].requirement_type:
        requirement_type_label = clauses[0].requirement_type

    # System prompt: persona/role
    system_prompt = """You are a senior regulatory compliance engineer specializing in multi-standard harmonization for e-bike and battery safety standards.

Your job:
- Take one cluster of closely related clauses from one or more safety standards.
- Understand the shared regulatory intent.
- Produce a consolidated requirement that is detailed, manual-ready text which can be pasted directly into a user manual or internal requirement specification.
- Preserve traceability to the original standards and clauses.
- Never weaken safety requirements or discard stricter conditions.
- Do not invent technical details or requirements that are not implied by the input clauses."""

    # User prompt: task with data
    user_prompt = f"""You are consolidating one pre-grouped cluster of similar clauses.

Cluster metadata:
- Regulatory category: {category_label}
- Requirement type: {requirement_type_label}

Source clauses (each line is one clause from a safety standard):

{clause_block}

Where each clause is formatted like:
"1. [UL 2271] 5.1.101: Full requirement text…"
"2. [EN 50604-1] 7.3.2: Full requirement text…"
"3. [IEC 62133-2] 8.2.1: Full requirement text…"

Your tasks:

1. Identify the shared REGULATORY INTENT:
   - What safety or compliance outcome are these clauses collectively trying to ensure?

2. Write ONE CONSOLIDATED REQUIREMENT that is:
   - Detailed and manual-ready: someone can copy-paste it into a user manual or requirement spec.
   - Structured: use a list format inside the text (e.g. a), b), c) or 1), 2), 3)) to enumerate all required elements.
   - Complete: include ALL mandatory elements and conditions that appear in any of the source clauses.
   - Precise: preserve all specific measurements, limits, temperatures, voltages, time durations, etc.
   - Conditional: preserve any "IF X THEN Y" logic explicitly.

   Avoid vague summaries like:
   - "Instructions must address stability."

   Prefer detailed, actionable text like:
   - "Instructions shall address safe transport, handling, and storage, including:
     a) Stability conditions during use, transportation, assembly, dismantling, testing, and foreseeable breakdowns
     b) Mass information for the machinery/product and component parts regularly transported separately
     c) Moving and storage procedures - IF moving or storage could result in damage creating risk of fire, electric shock, or injury during subsequent use, describe proper procedures preceded by warning statement
     d) Prevention of sudden movements or hazards due to instability when handled per instructions"

3. Identify and describe DIFFERENCES ACROSS STANDARDS:
   - Note differences in:
     - terminology,
     - thresholds or numeric limits,
     - required formats (e.g. paper vs digital),
     - requirement strength (e.g. "shall" vs "should"),
     - any standard-specific extra obligations.
   - Represent these as a list of structured entries per standard.

4. Identify UNIQUE REQUIREMENTS:
   - Requirements that appear ONLY in a single standard in this cluster and cannot be cleanly rolled into the consolidated requirement.
   - Summarize them briefly.

5. Identify CONFLICTS:
   - Only if there are true contradictions (statements that cannot both be true at the same time).
   - If there are no conflicts, return an empty string for conflicts.

VERY IMPORTANT CONSTRAINTS:
- Do NOT invent technical details or new requirements that are not implied by the input clauses.
- If a detail is ambiguous or missing in all clauses, do not guess.

OUTPUT FORMAT:

Return STRICT JSON ONLY using this EXACT schema:

{{
  "group_title": "Short human readable title for this cluster, e.g. 'Battery charging temperature limits'",
  "regulatory_intent": "One or two sentences describing the shared intent.",
  "consolidated_requirement": "One detailed, manual-ready, structured requirement string that can be pasted into a manual.",
  "differences_across_standards": [
    {{
      "standard_id": "e.g. 'UL 2271' or 'EN 50604-1'",
      "clause_labels": ["list", "of", "clause", "ids", "if known"],
      "difference_summary": "Short description of how this standard differs (e.g. stricter limits, additional notes, or different format)."
    }}
  ],
  "unique_requirements": "Short description of any unique requirements that cannot be fully merged into the consolidated requirement. Empty string if none.",
  "conflicts": "Short description of true conflicts between standards, or empty string if none."
}}

Constraints:
- The top-level value MUST be a single JSON object with exactly these keys.
- Do NOT include markdown, comments, or extra text outside JSON.
- Do NOT add any extra keys.
- If there are no differences, use an empty list for differences_across_standards.
- If there are no unique requirements or conflicts, use empty strings for those fields.
"""

    return system_prompt, user_prompt


def enrich_group_with_llm(
    clauses: List[Clause],
    group_index: int,
    call_llm: Callable[[str, str], str]
) -> RequirementGroup:
    """
    Use LLM to consolidate a group of clauses into a RequirementGroup with unified core requirement.

    HARD LOCK POLICY:
    Consolidation MUST be performed with Claude Sonnet 4.5.
    This ensures:
      - Deterministic JSON format
      - Consistent normative language
      - Stable consolidation logic across standards
      - Prevention of fallback hallucination behavior

    This function:
    1. Builds system + user prompts for the LLM
    2. Calls the LLM (via dependency injection with system + user messages)
    3. Parses the JSON response
    4. Creates a RequirementGroup object

    Args:
        clauses: List of Clause objects to consolidate
        group_index: Index of this group (used for group_id)
        call_llm: Function that takes (system_prompt, user_prompt) and returns LLM response string
                  MUST use Claude Sonnet 4.5

    Returns:
        RequirementGroup object with consolidated core requirement

    Raises:
        ValueError: If group is invalid (same-standard, empty, etc.)
        json.JSONDecodeError: If LLM response is not valid JSON
    """
    # Validate group
    if not clauses or len(clauses) < 2:
        raise ValueError(f"Group {group_index} must contain at least 2 clauses, got {len(clauses)}")

    standards = set(clause.standard_name for clause in clauses)
    if len(standards) < 2:
        raise ValueError(
            f"Group {group_index} contains clauses from only {standards}. "
            "RequirementGroup requires 2+ standards."
        )

    print(f"[CONSOLIDATE] Group {group_index}: Consolidating {len(clauses)} clauses from {len(standards)} standards")

    # Build system + user prompts
    system_prompt, user_prompt = build_llm_prompt_for_group(clauses, group_index)

    # Call LLM with system + user messages
    print(f"[CONSOLIDATE] Group {group_index}: Calling LLM...")
    llm_response = call_llm(system_prompt, user_prompt)
    print(f"[CONSOLIDATE] Group {group_index}: Received LLM response ({len(llm_response)} chars)")

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        if llm_response.strip().startswith('```'):
            # Extract content between code fences
            lines = llm_response.strip().split('\n')
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or not any(line.strip().startswith(c) for c in ['```']):
                    json_lines.append(line)
            llm_response = '\n'.join(json_lines)

        result = json.loads(llm_response)
    except json.JSONDecodeError as e:
        print(f"[CONSOLIDATE] Group {group_index}: ERROR - Failed to parse JSON response")
        print(f"[CONSOLIDATE] Response was: {llm_response[:500]}...")
        raise

    # Extract fields (new Phase 2 schema with differences_across_standards)
    group_title = result.get('group_title', '').strip()
    regulatory_intent = result.get('regulatory_intent', '').strip()
    consolidated_requirement = result.get('consolidated_requirement', '').strip()
    differences_across_standards = result.get('differences_across_standards', [])
    unique_requirements = result.get('unique_requirements', '').strip()
    conflicts = result.get('conflicts', '').strip()

    # Legacy fields (for backwards compatibility)
    core_requirement = consolidated_requirement or result.get('core_requirement', '').strip()
    analysis_notes = result.get('analysis_notes', regulatory_intent).strip()
    category = result.get('category', group_title).strip()

    if not core_requirement:
        raise ValueError(f"Group {group_index}: LLM did not provide a consolidated_requirement")

    # Create RequirementGroup with all Phase 2 fields
    group = RequirementGroup(
        group_id=group_index,
        core_requirement=core_requirement,
        clauses=clauses,
        group_title=group_title,
        regulatory_intent=regulatory_intent,
        consolidated_requirement=consolidated_requirement,
        differences_across_standards=differences_across_standards,
        unique_requirements=unique_requirements if unique_requirements else None,
        conflicts=conflicts if conflicts else None,
        analysis_notes=analysis_notes,
        category=category
    )

    # Use safe printing to handle Unicode characters (degree symbols, accents, etc.)
    try:
        print(f"[CONSOLIDATE] Group {group_index}: SUCCESS - Created RequirementGroup")
        print(f"  Title: {group_title}")
        print(f"  Regulatory Intent: {regulatory_intent[:80]}...")
        print(f"  Consolidated requirement: {consolidated_requirement[:100]}...")
        if conflicts:
            print(f"  ⚠️  CONFLICTS DETECTED: {conflicts[:80]}...")
    except UnicodeEncodeError:
        # Windows console encoding issue - print ASCII-safe version
        print(f"[CONSOLIDATE] Group {group_index}: SUCCESS - Created RequirementGroup")
        print(f"  Title: {group_title.encode('ascii', 'replace').decode('ascii')}")
        print(f"  Regulatory Intent: {regulatory_intent[:80].encode('ascii', 'replace').decode('ascii')}...")
        print(f"  Consolidated requirement: {consolidated_requirement[:100].encode('ascii', 'replace').decode('ascii')}...")
        if conflicts:
            print(f"  WARNING: CONFLICTS DETECTED: {conflicts[:80].encode('ascii', 'replace').decode('ascii')}...")

    return group


def consolidate_groups(
    all_clauses: List[Clause],
    groups: List[List[int]],
    call_llm: Callable[[str, str], str]
) -> List[RequirementGroup]:
    """
    Consolidate multiple groups using LLM enrichment.

    Args:
        all_clauses: Complete list of all Clause objects
        groups: List of groups (each group is a list of indices into all_clauses)
        call_llm: Function that takes (system_prompt, user_prompt) and returns LLM response string

    Returns:
        List of RequirementGroup objects with consolidated core requirements
    """
    requirement_groups = []

    print(f"[CONSOLIDATE] Processing {len(groups)} groups...")

    for i, group_indices in enumerate(groups):
        # Extract clauses for this group
        group_clauses = [all_clauses[idx] for idx in group_indices]

        # Skip extremely large groups that would exceed API limits
        # Empirically determined: prompts over ~100KB tend to fail with APIConnectionError
        # A group with 100 clauses typically generates ~100KB prompt
        MAX_CLAUSES_PER_GROUP = 100
        if len(group_clauses) > MAX_CLAUSES_PER_GROUP:
            print(f"[CONSOLIDATE] Group {i}: SKIPPING - Too large ({len(group_clauses)} clauses)")
            print(f"[CONSOLIDATE] Group {i}: Exceeds maximum of {MAX_CLAUSES_PER_GROUP} clauses per group")
            print(f"[CONSOLIDATE] Group {i}: Consider splitting this group or using a higher similarity threshold")
            continue

        try:
            # Use LLM to create consolidated RequirementGroup
            req_group = enrich_group_with_llm(group_clauses, i, call_llm)
            requirement_groups.append(req_group)
        except Exception as e:
            print(f"[CONSOLIDATE] Group {i}: ERROR - {e}")
            print(f"[CONSOLIDATE] Group {i}: Skipping this group")
            continue

    print(f"[CONSOLIDATE] Successfully consolidated {len(requirement_groups)}/{len(groups)} groups")
    return requirement_groups


# Stub LLM function for testing (can be replaced with real implementation)
def stub_call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Stub LLM function for testing. Returns a placeholder JSON response.

    Replace this with a real LLM call (e.g., using Anthropic Claude API) for production.
    """
    print("[CONSOLIDATE] WARNING: Using stub LLM function (not calling real LLM)")

    # Return a valid JSON response matching Phase 2 schema
    return json.dumps({
        "group_title": "Placeholder Title",
        "regulatory_intent": "This is a stub response. Implement real LLM to get actual consolidation.",
        "consolidated_requirement": "Placeholder consolidated requirement (LLM not implemented yet)",
        "differences_across_standards": [],
        "unique_requirements": "",
        "conflicts": ""
    })
