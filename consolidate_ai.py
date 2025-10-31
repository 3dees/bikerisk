"""
AI-powered consolidation logic for requirements.
Uses Claude (Anthropic) to analyze semantic similarity and suggest consolidations.
"""
import anthropic
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
    client = anthropic.Anthropic(api_key=api_key)

    # First pass: extract keywords and topics for each requirement
    print(f"[AI] Analyzing {len(requirements)} requirements...")

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

    for i, req1 in enumerate(requirements):
        if i in used_indices:
            continue

        group = [i]
        text1 = req1.get('Description', '') or req1.get('Requirement (Clause)', '')

        for j, req2 in enumerate(requirements[i+1:], start=i+1):
            if j in used_indices:
                continue

            text2 = req2.get('Description', '') or req2.get('Requirement (Clause)', '')

            # Check fuzzy similarity
            ratio = fuzz.token_sort_ratio(text1, text2) / 100.0

            if ratio >= threshold * 0.6:  # Lower threshold for pre-filtering
                group.append(j)
                used_indices.add(j)

        if len(group) >= 2:
            groups.append(group)
            for idx in group:
                used_indices.add(idx)

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
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        response_text = message.content[0].text

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

    except Exception as e:
        print(f"[AI ERROR] Failed to analyze group {group_id}: {e}")

    return None
