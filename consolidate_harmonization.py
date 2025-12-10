"""
Harmonization layer wrapper for Streamlit app.
Replaces consolidate_smart_ai.py with the TF-IDF clustering pipeline.
"""
import pandas as pd
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

from harmonization.models import Clause
from harmonization.grouping import group_clauses_by_category_then_similarity


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
    Use harmonization layer (TF-IDF clustering) to create cross-standard groups.
    
    Args:
        df: DataFrame with requirements (must have 'Standard/ Regulation' and 'Requirement (Clause)' columns)
        api_key: Anthropic API key (stored for future LLM consolidation)
        similarity_threshold: Base threshold for TF-IDF clustering (default 0.35)
        progress_callback: Optional function(message, progress_pct) for UI updates
    
    Returns:
        Dict with consolidation results compatible with Streamlit UI
    """
    
    if progress_callback:
        progress_callback("Converting DataFrame to Clause objects...", 5)
    
    # Step 1: Convert DataFrame to Clause objects
    all_clauses = []
    for idx, row in df.iterrows():
        standard = str(row.get('Standard/ Regulation', 'Unknown'))
        requirement_text = str(row.get('Requirement (Clause)', ''))
        clause_number = str(row.get('Clause', f'Req_{idx}'))
        
        if requirement_text and requirement_text.strip():
            clause = Clause(
                clause_number=clause_number,
                text=requirement_text,
                standard_name=standard
            )
            all_clauses.append(clause)
    
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
    
    try:
        group_indices, category_map = group_clauses_by_category_then_similarity(
            all_clauses,
            similarity_threshold=similarity_threshold
        )
    except Exception as e:
        print(f"[HARMONIZATION] Error during clustering: {e}")
        return {
            'groups': [],
            'ungrouped_indices': list(range(len(df))),
            'analysis_notes': f'Error during clustering: {str(e)}',
            'total_requirements': len(all_clauses),
            'grouped_count': 0,
            'ungrouped_count': len(all_clauses)
        }
    
    # Filter for cross-standard groups only
    cross_standard_groups = []
    for group_idx_list in group_indices:
        standards_in_group = set(all_clauses[i].standard_name for i in group_idx_list)
        if len(standards_in_group) >= 2:
            cross_standard_groups.append(group_idx_list)
    
    if not cross_standard_groups:
        # Return empty result if no cross-standard groups
        all_indices = list(range(len(df)))
        return {
            'groups': [],
            'ungrouped_indices': all_indices,
            'analysis_notes': f'Created {len(group_indices)} groups, but none were cross-standard. Try lowering the similarity threshold.',
            'total_requirements': len(all_clauses),
            'grouped_count': 0,
            'ungrouped_count': len(all_clauses)
        }
    
    if progress_callback:
        progress_callback(f"Found {len(cross_standard_groups)} cross-standard groups", 50)
    
    # Step 3: Convert to Streamlit-compatible format
    streamlit_groups = []
    grouped_indices = set()
    
    for group_idx, group_idx_list in enumerate(cross_standard_groups):
        # Get clauses in this group
        group_clauses = [all_clauses[i] for i in group_idx_list]
        grouped_indices.update(group_idx_list)
        
        # Extract standards
        standards = list(set(c.standard_name for c in group_clauses))
        
        # Build core requirement from first clause
        core_req = group_clauses[0].text if group_clauses else ""
        
        # Build critical differences from other clauses
        critical_differences = []
        for clause in group_clauses[1:]:
            critical_differences.append(f"{clause.standard_name} ({clause.clause_number}): {clause.text}")
        
        # Create ConsolidationGroup compatible with Streamlit UI
        streamlit_group = ConsolidationGroup(
            group_id=group_idx,
            topic=f"Group {group_idx + 1} - {', '.join(standards[:2])}",
            regulatory_intent="Cross-standard requirement cluster",
            core_requirement=core_req,
            applies_to_standards=standards,
            critical_differences=critical_differences,
            consolidation_potential=0.75,
            requirement_indices=group_idx_list,
            reasoning=f"TF-IDF similarity cluster with {len(group_idx_list)} requirements from {len(standards)} standards"
        )
        
        streamlit_groups.append(streamlit_group)
    
    # Ungrouped indices
    ungrouped_indices = list(set(range(len(all_clauses))) - grouped_indices)
    
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
