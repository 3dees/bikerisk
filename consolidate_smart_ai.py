"""
Smart AI consolidation based on regulatory intent and semantic understanding.
Uses Claude API with improved prompting for high-quality consolidations.
Includes automatic batching for large datasets.
"""

import anthropic
import httpx
import os
import json
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ConsolidationGroup:
    """A consolidation group with core requirement and differences"""
    group_id: int
    topic: str
    regulatory_intent: str
    core_requirement: str
    applies_to_standards: List[str]
    critical_differences: List[str]
    consolidation_potential: float  # 0.0-1.0
    requirement_indices: List[int]
    reasoning: str


def consolidate_with_smart_ai(
    df: pd.DataFrame,
    api_key: str,
    min_group_size: int = 3,
    max_group_size: int = 12,
    batch_size: int = 150  # Auto-batch if more than this
) -> Dict:
    """
    Use Claude to intelligently consolidate requirements based on regulatory intent.
    Automatically batches large datasets to avoid timeouts.
    
    Args:
        df: DataFrame with requirements
        api_key: Anthropic API key
        min_group_size: Minimum requirements per group (default 3)
        max_group_size: Maximum requirements per group (default 12)
        batch_size: Maximum requirements per batch (default 150)
    
    Returns:
        Dict with consolidation results
    """
    
    # Convert DataFrame to list format for Claude
    requirements = []
    for idx, row in df.iterrows():
        req_text = row.get('Requirement (Clause)', row.get('Description', ''))
        standard = row.get('Standard/ Regulation', row.get('Standard/Reg', ''))
        clause = row.get('Clause ID', row.get('Clause/Requirement', row.get('Clause', '')))
        
        if pd.notna(req_text) and str(req_text).strip():
            requirements.append({
                'index': idx,
                'text': str(req_text),
                'standard': str(standard) if pd.notna(standard) else 'Unknown',
                'clause': str(clause) if pd.notna(clause) else ''
            })
    
    total_requirements = len(requirements)
    print(f"[SMART AI] Processing {total_requirements} requirements")
    
    # Check if we need to batch
    if total_requirements > batch_size:
        print(f"[SMART AI] Dataset is large ({total_requirements} requirements)")
        print(f"[SMART AI] Using automatic batching ({batch_size} requirements per batch)")
        return _consolidate_batched(requirements, api_key, batch_size, min_group_size, max_group_size)
    else:
        print(f"[SMART AI] Dataset size OK - processing in single batch")
        return _consolidate_single_batch(requirements, api_key, min_group_size, max_group_size)


def _consolidate_single_batch(requirements: List[Dict], api_key: str, min_group_size: int = 3, max_group_size: int = 12) -> Dict:
    """Process a single batch of requirements."""
    
    # Setup client with long timeout
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com'
    
    try:
        http_client = httpx.Client(timeout=600.0)  # 10 minute timeout
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        print("[SMART AI] Client initialized with 10-minute timeout")
    except Exception as e:
        print(f"[SMART AI] Client init with http_client failed: {e}, using simple init")
        client = anthropic.Anthropic(api_key=api_key)
    
    print(f"[SMART AI] Estimated time: 3-8 minutes for {len(requirements)} requirements")
    
    # Build comprehensive prompt
    prompt = f"""You are an expert in e-bike and bicycle safety standards and compliance requirements.

You have {len(requirements)} requirements from various standards (EN, CFR, UL, MD, etc.).

YOUR TASK: Identify groups of requirements that share the SAME REGULATORY INTENT - meaning they're all trying to ensure the same safety/compliance outcome, even if worded differently.

REQUIREMENTS:
"""
    
    for req in requirements:
        prompt += f"\n[{req['index']}] {req['standard']} (Clause {req['clause']}): {req['text'][:200]}"
        if len(req['text']) > 200:
            prompt += "..."
    
    prompt += f"""

CRITICAL INSTRUCTIONS:

1. **Group by Regulatory Intent:** Group requirements that achieve the SAME compliance goal:
   - "Provide assembly instructions" (regardless of exact wording)
   - "Specify battery charging temperature" (even if different standards use different formats)
   - "Label which brake controls which wheel" (same intent across standards)

2. **Respect Compliance Keywords:** Pay close attention to:
   - "shall" vs "must" vs "should" (different requirement levels)
   - "instructions" vs "user manual" vs "documentation" (same thing - treat as equivalent)
   - "included with product" vs "made available" (different obligations)

3. **Group Size:** Create groups of {min_group_size}-{max_group_size} requirements. Avoid groups smaller than {min_group_size} or larger than {max_group_size}.

4. **CRITICAL: Create DETAILED Core Requirements**
   
   ❌ DO NOT create vague summaries like: "Instructions must address stability"
   
   ✅ DO create detailed, structured requirements listing ALL elements:
   
   Example Format:
   "Instructions shall address safe transport, handling, and storage, including:
   
   a) Stability conditions during use, transportation, assembly, dismantling, testing, and foreseeable breakdowns
   
   b) Mass information for the machinery/product and component parts regularly transported separately
   
   c) Moving and storage procedures - IF moving or storage could result in damage creating risk of fire, electric shock, or injury during subsequent use, describe proper procedures preceded by warning statement
   
   d) Prevention of sudden movements or hazards due to instability when handled per instructions"
   
   **Key Rules for Core Requirements:**
   - Use structured format with bullets (a, b, c) or numbers (1, 2, 3)
   - List EVERY specific element that must be included
   - Consolidate where standards say the EXACT same thing
   - Preserve ALL conditional clauses (IF X, THEN Y)
   - Keep ALL measurements, temperatures, voltages, specifications
   - Make it ACTIONABLE - someone should be able to write the manual section using this

5. **Preserve Critical Differences:** For each group, explicitly list:
   - Different temperature/voltage/measurement specifications
   - Format requirements (paper vs digital)
   - Different legal obligations (shall vs may)
   - Standard-specific details that DON'T fit in the consolidated core

OUTPUT FORMAT (JSON):
{{
  "groups": [
    {{
      "group_id": 0,
      "topic": "Brief topic name",
      "regulatory_intent": "What this group of requirements is trying to achieve",
      "core_requirement": "DETAILED, structured requirement with all elements listed using a), b), c) format",
      "applies_to_standards": ["16 CFR Part 1512", "EN 15194", ...],
      "critical_differences": [
        "16 CFR requires paper format only",
        "EN 15194 allows digital with paper",
        "UL 2849 specifies temperature range 0-40°C while EN specifies -20 to 50°C"
      ],
      "consolidation_potential": 0.85,
      "requirement_indices": [0, 5, 12, 23],
      "reasoning": "Why these requirements share the same regulatory intent"
    }}
  ],
  "ungrouped_indices": [3, 7, 15],
  "analysis_notes": "Overall observations about the requirements"
}}

CONSOLIDATION POTENTIAL SCORING:
- 0.9-1.0: Nearly identical, easy to consolidate
- 0.7-0.89: Same intent, some differences to preserve
- 0.5-0.69: Related intent, significant differences
- Below 0.5: Don't group

Only create groups with consolidation_potential >= 0.6

**REMEMBER:** The core_requirement should be detailed enough that someone could write the actual manual section from it. It's NOT just a summary - it's the consolidated instruction text itself!

Be thorough and detailed. Create consolidations that help reduce manual size while preserving ALL critical compliance details.
"""
    
    try:
        print(f"[SMART AI] Sending to Claude Sonnet 4.5...")
        
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            temperature=0,
            timeout=600.0,  # 10 minute timeout
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        print(f"[SMART AI] Received response ({len(response_text)} chars)")
        
        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
        
        result = json.loads(json_str)
        
        # Convert to ConsolidationGroup objects
        groups = []
        for g in result.get('groups', []):
            group = ConsolidationGroup(
                group_id=g['group_id'],
                topic=g['topic'],
                regulatory_intent=g['regulatory_intent'],
                core_requirement=g['core_requirement'],
                applies_to_standards=g['applies_to_standards'],
                critical_differences=g['critical_differences'],
                consolidation_potential=g['consolidation_potential'],
                requirement_indices=g['requirement_indices'],
                reasoning=g['reasoning']
            )
            groups.append(group)
        
        print(f"[SMART AI] Created {len(groups)} consolidation groups")
        
        return {
            'groups': groups,
            'ungrouped_indices': result.get('ungrouped_indices', []),
            'analysis_notes': result.get('analysis_notes', ''),
            'total_requirements': len(requirements),
            'grouped_count': sum(len(g.requirement_indices) for g in groups),
            'ungrouped_count': len(result.get('ungrouped_indices', []))
        }
        
    except anthropic.APITimeoutError as e:
        print(f"[SMART AI TIMEOUT] {e}")
        raise ValueError(
            f"Analysis timed out after 10 minutes. Your dataset has {len(requirements)} requirements. "
            f"Try using a smaller batch size or contact support. "
            f"Original error: {str(e)}"
        )
    except Exception as e:
        print(f"[SMART AI ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Smart AI consolidation failed: {str(e)}")


def _consolidate_batched(requirements: List[Dict], api_key: str, batch_size: int, min_group_size: int = 3, max_group_size: int = 12) -> Dict:
    """
    Process requirements in batches and combine results.
    Useful for very large datasets (150+ requirements).
    """
    total_requirements = len(requirements)
    num_batches = (total_requirements + batch_size - 1) // batch_size
    
    print(f"[BATCH] Splitting {total_requirements} requirements into {num_batches} batches")
    
    all_groups = []
    all_ungrouped = []
    batch_analyses = []
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_requirements)
        batch_reqs = requirements[start_idx:end_idx]

        # Skip tiny batches (too small to consolidate meaningfully)
        if len(batch_reqs) < 5:
            print(f"\n[BATCH {batch_num + 1}/{num_batches}] ⚠️ Skipping (only {len(batch_reqs)} requirements - too small to consolidate)")
            # Add them to ungrouped
            for req in batch_reqs:
                all_ungrouped.append(req['index'])
            continue

        print(f"\n[BATCH {batch_num + 1}/{num_batches}] Processing requirements {start_idx}-{end_idx-1}")

        try:
            batch_result = _consolidate_single_batch(batch_reqs, api_key, min_group_size, max_group_size)

            # Verify groups were created
            if not batch_result.get('groups') or len(batch_result['groups']) == 0:
                print(f"[BATCH {batch_num + 1}/{num_batches}] ⚠️ No groups created")
                # Add batch requirements to ungrouped
                for req in batch_reqs:
                    all_ungrouped.append(req['index'])
                continue

            # Adjust group IDs to be globally unique
            for group in batch_result['groups']:
                group.group_id = len(all_groups)
                all_groups.append(group)

            # Adjust ungrouped indices to be globally correct
            for idx in batch_result.get('ungrouped_indices', []):
                all_ungrouped.append(batch_reqs[idx]['index'])

            batch_analyses.append(batch_result.get('analysis_notes', ''))

            print(f"[BATCH {batch_num + 1}/{num_batches}] ✓ Created {len(batch_result['groups'])} groups")

        except IndexError as e:
            print(f"[BATCH {batch_num + 1}/{num_batches}] ✗ IndexError: {e}")
            print(f"[BATCH {batch_num + 1}/{num_batches}]   Batch size: {len(batch_reqs)}")
            print(f"[BATCH {batch_num + 1}/{num_batches}]   Continuing with next batch...")
            # Add batch requirements to ungrouped
            for req in batch_reqs:
                all_ungrouped.append(req['index'])
            continue
        except Exception as e:
            print(f"[BATCH {batch_num + 1}/{num_batches}] ✗ Unexpected error: {e}")
            print(f"[BATCH {batch_num + 1}/{num_batches}]   Continuing with next batch...")
            # Add batch requirements to ungrouped
            for req in batch_reqs:
                all_ungrouped.append(req['index'])
            continue
    
    combined_analysis = f"Processed {num_batches} batches. " + " | ".join(batch_analyses)
    
    print(f"\n[BATCH] Complete! Created {len(all_groups)} total groups from {num_batches} batches")
    
    return {
        'groups': all_groups,
        'ungrouped_indices': all_ungrouped,
        'analysis_notes': combined_analysis,
        'total_requirements': total_requirements,
        'grouped_count': sum(len(g.requirement_indices) for g in all_groups),
        'ungrouped_count': len(all_ungrouped)
    }