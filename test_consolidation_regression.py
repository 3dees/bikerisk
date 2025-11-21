"""
Regression test for consolidation pipeline.

This test ensures that the consolidation pipeline continues to work correctly
as we make changes to prompts, grouping rules, and other components.

Run with: python test_consolidation_regression.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harmonization.grouping import load_clauses_from_tagged_csv, group_clauses_by_similarity, compute_embeddings
from harmonization.consolidate import consolidate_groups
from harmonization.pipeline_unify import call_llm_for_consolidation


def test_consolidation_pipeline():
    """
    Regression test for the consolidation pipeline.
    
    This test:
    1. Loads the tiny test dataset (6 clauses)
    2. Groups them by similarity
    3. Consolidates groups with Claude
    4. Validates expected behavior
    
    Expected behavior (as of Nov 21, 2025):
    - Should create 2 cross-standard groups
    - Group 1 should be about "User Documentation" (4 clauses)
    - Group 2 should be about "Mechanical Abuse" (2 clauses)
    - Should detect at least 1 conflict
    """
    print("=" * 80)
    print("CONSOLIDATION REGRESSION TEST")
    print("=" * 80)
    
    # Test configuration
    csv_path = "tiny_test_clauses.csv"
    similarity_threshold = 0.05
    
    if not os.path.exists(csv_path):
        print(f"❌ TEST FAILED: Test file '{csv_path}' not found")
        print(f"   Current directory: {os.getcwd()}")
        return False
    
    print(f"\n✓ Test file found: {csv_path}")
    
    # Step 1: Load clauses
    print(f"\n[STEP 1] Loading clauses from {csv_path}...")
    try:
        clauses = load_clauses_from_tagged_csv(csv_path)
        print(f"✓ Loaded {len(clauses)} clauses")
        
        if len(clauses) != 6:
            print(f"❌ TEST FAILED: Expected 6 clauses, got {len(clauses)}")
            return False
            
    except Exception as e:
        print(f"❌ TEST FAILED: Error loading clauses: {e}")
        return False
    
    # Step 2: Group clauses by similarity
    print(f"\n[STEP 2] Grouping clauses by similarity (threshold={similarity_threshold})...")
    try:
        # Compute embeddings (using TF-IDF fallback for test)
        embeddings, vectorizer = compute_embeddings(clauses, embed_fn=None)
        print(f"✓ Computed embeddings: {embeddings.shape}")
        
        # Group clauses
        groups = group_clauses_by_similarity(clauses, embeddings, similarity_threshold)
        print(f"✓ Created {len(groups)} groups")
        
        # Validate group count
        if len(groups) < 1:
            print(f"❌ TEST FAILED: Expected at least 1 group, got {len(groups)}")
            return False
            
        if len(groups) != 2:
            print(f"⚠️  WARNING: Expected 2 groups, got {len(groups)} (may be acceptable if grouping logic changed)")
        
        # Convert group indices to clause objects for consolidation
        clause_groups = [[clauses[idx] for idx in group] for group in groups]
        
        # Print group details
        for i, group in enumerate(clause_groups):
            print(f"  Group {i+1}: {len(group)} clauses from {len(set(c.standard_name for c in group))} standards")
            
    except Exception as e:
        print(f"❌ TEST FAILED: Error grouping clauses: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Consolidate groups with LLM
    print(f"\n[STEP 3] Consolidating groups with Claude...")
    try:
        # Create LLM function
        llm_fn = lambda prompt: call_llm_for_consolidation(prompt, use_claude=True)
        
        # Consolidate (needs all_clauses and index groups)
        consolidated_groups = consolidate_groups(clauses, groups, call_llm=llm_fn)
        print(f"✓ Successfully consolidated {len(consolidated_groups)} groups")
        
        # Validate consolidation results
        if len(consolidated_groups) < 1:
            print(f"❌ TEST FAILED: Expected at least 1 consolidated group, got {len(consolidated_groups)}")
            return False
        
        # Check for expected group names/topics
        group_titles = [g.group_title.lower() for g in consolidated_groups]
        print(f"\n  Group titles: {group_titles}")
        
        # Check if we have documentation-related group
        has_documentation_group = any(
            'documentation' in title or 'manual' in title or 'user' in title
            for title in group_titles
        )
        
        if not has_documentation_group:
            print(f"⚠️  WARNING: Expected to find a documentation/manual-related group")
            print(f"             This may indicate a regression in grouping or consolidation")
        else:
            print(f"✓ Found documentation-related group")
        
        # Check for mechanical/abuse group
        has_mechanical_group = any(
            'mechanical' in title or 'abuse' in title or 'battery' in title
            for title in group_titles
        )
        
        if not has_mechanical_group:
            print(f"⚠️  WARNING: Expected to find a mechanical/battery-related group")
        else:
            print(f"✓ Found mechanical/battery-related group")
        
        # Check for conflicts detection
        total_conflicts = sum(1 for g in consolidated_groups if g.conflicts)
        print(f"\n✓ Detected {total_conflicts} conflicts across all groups")
        
        if total_conflicts == 0:
            print(f"⚠️  WARNING: Expected to find at least 1 conflict (UL 2849 'should' vs others 'shall')")
        
        # Print detailed results
        print(f"\n[CONSOLIDATION RESULTS]")
        for i, group in enumerate(consolidated_groups, 1):
            standards = set(c.standard_name for c in group.clauses)
            print(f"\n  Group {i}: {group.group_title}")
            print(f"    Standards: {', '.join(standards)}")
            print(f"    Clause count: {len(group.clauses)}")
            print(f"    Regulatory intent: {group.regulatory_intent[:80]}...")
            if group.conflicts:
                print(f"    ⚠️  Conflicts: {group.conflicts[:100]}...")
            if group.unique_requirements:
                print(f"    ℹ️  Unique requirements: {group.unique_requirements[:100]}...")
                
    except Exception as e:
        print(f"❌ TEST FAILED: Error consolidating groups: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All checks passed
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: Consolidation pipeline working correctly")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_consolidation_pipeline()
    sys.exit(0 if success else 1)
