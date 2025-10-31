"""
AI-powered consolidation logic for requirements.
Uses Claude (Anthropic) to analyze semantic similarity and suggest consolidations.
"""
import anthropic
import httpx
import os
from typing import List, Dict, Tuple
import pandas as pd
from rapidfuzz import fuzz


def analyze_similarity_with_ai(
    requirements: List[Dict],
    api_key: str,
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Analyze requirements using Claude AI to find similar ones that can be consolidated.

    Args:
        requirements: List of requirement dicts with keys like 'Description', 'Standard/Reg', etc.
        api_key: Anthropic API key
        similarity_threshold: Threshold for grouping (0-1)

    Returns:
        List of consolidation suggestions, each with:
        - 'group_id': unique ID for this consolidation group
        - 'original_requirements': list of original requirement indices
        - 'similarity_score': AI-assessed similarity (0-1)
        - 'reasoning': why they should be consolidated
        - 'suggested_consolidation': AI-generated merged text
        - 'topic_keywords': extracted topic keywords
    """
    # Setup client with proxy bypass (similar to extract_ai.py)
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(proxies=None, timeout=120.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        print("[AI] Client initialized with proxy bypass")
    except Exception as e:
        print(f"[AI] Client init error: {e}, trying without custom http_client")
        client = anthropic.Anthropic(api_key=api_key)

    # First pass: extract keywords and topics for each requirement
    print(f"[AI] Analyzing {len(requirements)} requirements...")

    # If threshold is very low (< 0.1), send everything to AI in small groups
    if similarity_threshold < 0.1:
        print(f"[AI] Low threshold ({similarity_threshold}), creating groups of 3-4 for AI analysis...")
        potential_groups = []
        # Create groups of 3-4 items for AI to analyze
        for i in range(0, len(requirements), 3):
            group_indices = list(range(i, min(i+4, len(requirements))))
            if len(group_indices) >= 2:
                potential_groups.append(group_indices)
    else:
        # Group requirements by topic using rapid fuzzy matching first
        # This reduces the number of AI calls needed
        potential_groups = _find_potential_groups_fuzzy(requirements, similarity_threshold)

    print(f"[AI] Found {len(potential_groups)} potential groups to analyze with AI...")

    consolidations = []

    for group_idx, group_indices in enumerate(potential_groups):
        if len(group_indices) < 2:
            continue  # Skip single-item groups

        # Get the actual requirements for this group
        group_reqs = [requirements[i] for i in group_indices]

        # Ask Claude to analyze this group
        consolidation = _analyze_group_with_claude(
            client,
            group_reqs,
            group_indices,
            group_idx
        )

        if consolidation and consolidation['similarity_score'] >= similarity_threshold:
            consolidations.append(consolidation)

    print(f"[AI] Generated {len(consolidations)} consolidation suggestions")
    return consolidations


def _find_potential_groups_fuzzy(
    requirements: List[Dict],
    threshold: float
) -> List[List[int]]:
    """
    Use fuzzy matching to find potential groups before expensive AI analysis.
    This is a pre-filter to reduce AI API calls.
    """
    groups = []
    used_indices = set()

    # Use a very low fuzzy threshold - we want AI to do the real analysis
    # If user sets threshold to 0, we want to check everything
    fuzzy_threshold = max(0.3, threshold * 0.5)  # Minimum 30% similarity for pre-filter

    print(f"[FUZZY] Using fuzzy threshold: {fuzzy_threshold:.2f} (user threshold: {threshold:.2f})")

    for i, req1 in enumerate(requirements):
        if i in used_indices:
            continue

        group = [i]
        text1 = req1.get('Description', '') or req1.get('Requirement (Clause)', '')

        # Handle NaN/None values
        if pd.isna(text1) or not text1:
            continue
        text1 = str(text1)

        standard1 = req1.get('Standard/Reg', '') or req1.get('Standard/ Regulation', '')
        if pd.isna(standard1):
            standard1 = '[Unknown]'

        for j, req2 in enumerate(requirements[i+1:], start=i+1):
            if j in used_indices:
                continue

            text2 = req2.get('Description', '') or req2.get('Requirement (Clause)', '')

            # Handle NaN/None values
            if pd.isna(text2) or not text2:
                continue
            text2 = str(text2)

            standard2 = req2.get('Standard/Reg', '') or req2.get('Standard/ Regulation', '')
            if pd.isna(standard2):
                standard2 = '[Unknown]'

            # Check fuzzy similarity
            ratio = fuzz.token_sort_ratio(text1, text2) / 100.0

            if ratio >= fuzzy_threshold:
                group.append(j)
                used_indices.add(j)

                # Log if this is a cross-standard match
                if standard1 != standard2:
                    print(f"[FUZZY] Cross-standard match! {standard1[:30]} vs {standard2[:30]} (ratio: {ratio:.2f})")

        if len(group) >= 2:
            groups.append(group)
            for idx in group:
                used_indices.add(idx)

            # Show standards in this group
            standards_in_group = set()
            for idx in group:
                std = requirements[idx].get('Standard/Reg', '') or requirements[idx].get('Standard/ Regulation', '')
                if not pd.isna(std):
                    standards_in_group.add(str(std)[:30])  # Truncate for readability

            print(f"[FUZZY] Found group of {len(group)} items with {len(standards_in_group)} unique standards: {list(standards_in_group)}")

    print(f"[FUZZY] Total groups found: {len(groups)}")
    return groups


def _analyze_group_with_claude(
    client: anthropic.Anthropic,
    requirements: List[Dict],
    indices: List[int],
    group_id: int
) -> Dict:
    """
    Use Claude to analyze a group of potentially similar requirements.
    """
    # Format requirements for Claude
    req_texts = []
    for i, req in enumerate(requirements):
        text = req.get('Description', '') or req.get('Requirement (Clause)', '')
        standard = req.get('Standard/Reg', '') or req.get('Standard/ Regulation', '')
        clause = req.get('Clause/Requirement', '') or req.get('Clause', '')
        scope = req.get('Requirement scope', '') or req.get('Requirement Scope', '')

        req_texts.append(f"[{i+1}] Standard: {standard}, Clause: {clause}\nScope: {scope}\nText: {text}")

    prompt = f"""You are analyzing requirements from e-bike safety standards to suggest consolidations.

Here are {len(requirements)} potentially similar requirements:

{chr(10).join(req_texts)}

Task:
1. Analyze if these requirements have the SAME REGULATORY INTENT and TOPIC
2. Determine if they can be safely consolidated without losing vital information
3. Generate a consolidated version that preserves ALL critical details:
   - Specific measurements, voltages, temperatures, etc.
   - Legal keywords (shall, must, required)
   - Safety warnings and cautions
   - Clause references
   - Any unique requirements from each source

Respond in JSON format:
{{
  "can_consolidate": true/false,
  "similarity_score": 0.0-1.0,
  "topic_keywords": ["keyword1", "keyword2"],
  "reasoning": "Why these should/shouldn't be consolidated",
  "suggested_consolidation": "The merged requirement text (or null if can't consolidate)",
  "critical_differences": ["any vital differences that must be preserved"]
}}

IMPORTANT: Only suggest consolidation if the regulatory intent is IDENTICAL and no vital information would be lost."""

    try:
        print(f"[AI] Analyzing group {group_id} ({len(requirements)} requirements)...")
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=2000,
            timeout=60.0,  # 60 second timeout
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        response_text = message.content[0].text
        print(f"[AI] Group {group_id} analysis complete")

        # Parse JSON response
        import json
        # Try to extract JSON from the response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)

        if result.get('can_consolidate'):
            return {
                'group_id': group_id,
                'original_requirements': indices,
                'similarity_score': result.get('similarity_score', 0.0),
                'reasoning': result.get('reasoning', ''),
                'suggested_consolidation': result.get('suggested_consolidation', ''),
                'topic_keywords': result.get('topic_keywords', []),
                'critical_differences': result.get('critical_differences', [])
            }
        else:
            print(f"[AI] Group {group_id} cannot be consolidated: {result.get('reasoning', 'No reason given')}")

    except Exception as e:
        print(f"[AI ERROR] Failed to analyze group {group_id}: {e}")
        import traceback
        traceback.print_exc()

    return None
