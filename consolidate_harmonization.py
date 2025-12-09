"""
Harmonization layer wrapper for Streamlit app.
Replaces consolidate_smart_ai.py with the TF-IDF + LLM consolidation pipeline.
"""
import pandas as pd
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import tempfile
from pathlib import Path

from harmonization.grouping import load_clauses_from_dataframe, group_clauses_by_category_then_similarity
from harmonization.consolidate import consolidate_groups


@dataclass
class ConsolidationGroup:
    """Compatibility wrapper for Streamlit UI expectations"""
    group_id: int
    topic: str
    regulatory_intent: str
    core_requirement: str
    applies_to_standards: List[str]
    critical_differences: List[str]
    consolidation_potential: float
    requirement_indices: List[int]
    reasoning: str


def consolidate_with_harmonization(
    df: pd.DataFrame,
    api_key: str,
    similarity_threshold: float = 0.35,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Use harmonization layer (TF-IDF clustering + LLM consolidation) to create cross-standard groups.
    
    Args:
        df: DataFrame with requirements (must have Standard/Reg and Description columns)
        api_key: Anthropic API key (not used - harmonization uses environment variable)
        similarity_threshold: Base threshold for TF-IDF clustering (default 0.35)
        progress_callback: Optional function(message, progress_pct) for UI updates
    
    Returns:
        Dict with consolidation results compatible with Streamlit UI
    """
    
    if progress_callback:
        progress_callback("Loading requirements...", 5)
    
    # Step 1: Load clauses from DataFrame
    all_clauses = load_clauses_from_dataframe(df)
    
    if not all_clauses:
        return {
            'groups': [],
            'ungrouped_indices': [],
            'analysis_notes': 'No valid requirements found in DataFrame',
            'total_requirements': 0,
            'grouped_count': 0,
            'ungrouped_count': 0
        }
    
    if progress_callback:
        progress_callback(f"Loaded {len(all_clauses)} requirements", 10)
    
    # Step 2: Group by category + similarity (TF-IDF clustering)
    if progress_callback:
        progress_callback("Clustering requirements by similarity...", 20)
    
    groups = group_clauses_by_category_then_similarity(
        all_clauses,
        similarity_threshold=similarity_threshold
    )
    
    # Filter for cross-standard groups only
    cross_standard_groups = [g for g in groups if len(set(c.standard_id for c in g.clauses)) >= 2]
    
    if not cross_standard_groups:
        # Return empty result if no cross-standard groups
        all_indices = [c.original_index for c in all_clauses]
        return {
            'groups': [],
            'ungrouped_indices': all_indices,
            'analysis_notes': f'Created {len(groups)} groups, but none were cross-standard. Try lowering the similarity threshold.',
            'total_requirements': len(all_clauses),
            'grouped_count': 0,
            'ungrouped_count': len(all_clauses)
        }
    
    if progress_callback:
        progress_callback(f"Found {len(cross_standard_groups)} cross-standard groups", 30)
    
    # Step 3: Consolidate each group with LLM
    if progress_callback:
        progress_callback("Consolidating groups with Claude...", 40)
    
    consolidated_groups = []
    total_groups = len(cross_standard_groups)
    
    # Skip groups that are too large (>100 clauses)
    MAX_GROUP_SIZE = 100
    
    for idx, group in enumerate(cross_standard_groups):
        if len(group.clauses) > MAX_GROUP_SIZE:
            print(f"[HARMONIZATION] Skipping group {idx} - too large ({len(group.clauses)} clauses)")
            continue
        
        # Update progress
        progress_pct = 40 + int((idx / total_groups) * 50)
        if progress_callback:
            progress_callback(f"Consolidating group {idx + 1}/{total_groups}...", progress_pct)
        
        try:
            # Consolidate with LLM
            result = consolidate_groups([group])
            
            if result and len(result) > 0:
                consolidated_groups.append(result[0])
        except Exception as e:
            print(f"[HARMONIZATION] Error consolidating group {idx}: {e}")
            continue
    
    if progress_callback:
        progress_callback("Finalizing results...", 95)
    
    # Step 4: Convert to Streamlit-compatible format
    streamlit_groups = []
    grouped_indices = set()
    
    for idx, cons_group in enumerate(consolidated_groups):
        # Extract requirement indices
        req_indices = [c.original_index for c in cons_group.clauses]
        grouped_indices.update(req_indices)
        
        # Extract standards
        standards = list(set(c.standard_id for c in cons_group.clauses))
        
        # Build critical differences list
        critical_differences = []
        for diff in cons_group.differences_across_standards:
            std_id = diff.standard_id
            diff_summary = diff.difference_summary
            clause_labels = ', '.join(diff.clause_labels)
            critical_differences.append(f"{std_id} ({clause_labels}): {diff_summary}")
        
        # Add unique requirements
        if cons_group.unique_requirements:
            for unique_req in cons_group.unique_requirements:
                critical_differences.append(f"UNIQUE: {unique_req}")
        
        # Add conflicts
        if cons_group.conflicts:
            for conflict in cons_group.conflicts:
                critical_differences.append(f"CONFLICT: {conflict}")
        
        # Create ConsolidationGroup compatible with Streamlit UI
        streamlit_group = ConsolidationGroup(
            group_id=idx,
            topic=cons_group.group_title or f"Group {idx + 1}",
            regulatory_intent=cons_group.regulatory_intent or "",
            core_requirement=cons_group.consolidated_requirement or "",
            applies_to_standards=standards,
            critical_differences=critical_differences,
            consolidation_potential=0.85,  # Placeholder - harmonization doesn't compute this
            requirement_indices=req_indices,
            reasoning=f"Consolidated from {len(req_indices)} requirements across {len(standards)} standards"
        )
        
        streamlit_groups.append(streamlit_group)
    
    # Ungrouped indices
    all_indices = set(c.original_index for c in all_clauses)
    ungrouped_indices = list(all_indices - grouped_indices)
    
    if progress_callback:
        progress_callback("Complete!", 100)
    
    return {
        'groups': streamlit_groups,
        'ungrouped_indices': ungrouped_indices,
        'analysis_notes': f'Harmonization complete: {len(streamlit_groups)} cross-standard groups created',
        'total_requirements': len(all_clauses),
        'grouped_count': len(grouped_indices),
        'ungrouped_count': len(ungrouped_indices)
    }
