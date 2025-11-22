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
) -> str:
    """
    Build a SAFE, Sonnet-4.5-optimized prompt for consolidating cross-standard
    requirement groups. This version:
    - Uses standard_name (not standard_id per patch, but we use what's available)
    - Avoids markdown formatting
    - Dynamically generates JSON schema per group
    - Prevents hallucinations and invented technical details
    - Forces strict JSON output only

    Args:
        clauses: List of similar Clause objects to consolidate
        group_index: Index of this group (for logging/tracking)

    Returns:
        Formatted prompt string for the LLM
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

    # Dynamically generate differences schema for the exact standards in this group
    diff_items = []
    for st in unique_standards:
        diff_items.append(f'{{"standard": "{st}", "differences": ""}}')
    differences_schema = ",\n      ".join(diff_items)

    # Determine category hint (optional)
    category_hint = ""
    if clauses and clauses[0].category:
        category = clauses[0].category
        category_title = CATEGORY_TITLE_MAP.get(category, category)
        category_hint = f"\nREGULATORY CATEGORY: {category_title}\nUse this category as guidance when choosing the group_title.\n"

    prompt = f"""You are a senior regulatory compliance engineer specializing in multi-standard harmonization.
Your task is to consolidate a group of similar clauses from different standards into one unified requirement.
{category_hint}
Source clauses:
{clause_block}

Follow these rules carefully:
- Identify the shared regulatory intent across all clauses.
- Produce a detailed consolidated requirement containing all mandatory elements.
- Do NOT invent technical details. Use only information present in the clauses.
- Identify differences per standard (terminology, thresholds, extra requirements).
- Identify unique requirements (appear in only one standard).
- Identify conflicts (statements that cannot both be true).
- If no unique requirements or conflicts exist, return "" for those fields.
- Output STRICT JSON ONLY.

Provide JSON using this EXACT schema:
{{
  "group_title": "",
  "regulatory_intent": "",
  "consolidated_requirement": "",
  "differences_across_standards": [
      {differences_schema}
  ],
  "unique_requirements": "",
  "conflicts": ""
}}

Return JSON only."""

    return prompt


def enrich_group_with_llm(
    clauses: List[Clause],
    group_index: int,
    call_llm: Callable[[str], str]
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
    1. Builds a prompt for the LLM
    2. Calls the LLM (via dependency injection)
    3. Parses the JSON response
    4. Creates a RequirementGroup object

    Args:
        clauses: List of Clause objects to consolidate
        group_index: Index of this group (used for group_id)
        call_llm: Function that takes a prompt string and returns LLM response string
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

    # Build prompt
    prompt = build_llm_prompt_for_group(clauses, group_index)

    # Call LLM
    print(f"[CONSOLIDATE] Group {group_index}: Calling LLM...")
    llm_response = call_llm(prompt)
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

    # Extract fields (new Phase 2 schema)
    group_title = result.get('group_title', '').strip()
    regulatory_intent = result.get('regulatory_intent', '').strip()
    consolidated_requirement = result.get('consolidated_requirement', '').strip()
    differences = result.get('differences_across_standards', [])
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
        differences=differences,
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
    call_llm: Callable[[str], str]
) -> List[RequirementGroup]:
    """
    Consolidate multiple groups using LLM enrichment.

    Args:
        all_clauses: Complete list of all Clause objects
        groups: List of groups (each group is a list of indices into all_clauses)
        call_llm: Function that takes a prompt string and returns LLM response string

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
def stub_call_llm(prompt: str) -> str:
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
