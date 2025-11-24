"""
Test consolidation on smaller groups only (skip Group 0 mega-group).
"""
import time
from harmonization.grouping import load_clauses_from_tagged_csv, group_clauses_by_category_then_similarity
from harmonization.consolidate import consolidate_groups
from harmonization.pipeline_unify import call_llm_for_consolidation
from harmonization.report_builder import save_html_report

def test_small_groups_only():
    """Test Phase 2 consolidation on groups 1-7 (skip 0)."""
    print("=" * 80)
    print("TESTING SMALL GROUPS (1-7) - Skip Group 0")
    print("=" * 80)

    # Load and group
    all_clauses = load_clauses_from_tagged_csv("all3.csv")
    groups, category_map = group_clauses_by_category_then_similarity(
        all_clauses,
        similarity_threshold=0.05,
        embed_fn=None
    )

    print(f"\nFound {len(groups)} groups total")

    # Skip Group 0 (too large), test Groups 1-7
    groups_to_test = groups[1:]  # Skip index 0

    print(f"\nTesting {len(groups_to_test)} groups (Groups 1-{len(groups)-1})")
    for i, group_indices in enumerate(groups_to_test, start=1):
        group_clauses = [all_clauses[idx] for idx in group_indices]
        print(f"  Group {i}: {len(group_clauses)} clauses")

    # Create LLM function
    llm_fn = lambda sys_prompt, usr_prompt: call_llm_for_consolidation(sys_prompt, usr_prompt, use_claude=True)

    # Consolidate with retries
    print("\n" + "=" * 80)
    print("Starting consolidation...")
    print("=" * 80)

    requirement_groups = consolidate_groups(all_clauses, groups_to_test, llm_fn)

    print(f"\n[TEST] Successfully consolidated {len(requirement_groups)}/{len(groups_to_test)} groups")

    if requirement_groups:
        output_path = "all3_small_groups_consolidation.html"
        print(f"\n[TEST] Generating HTML report: {output_path}")
        save_html_report(requirement_groups, output_path)
        print(f"[TEST] Report saved successfully")
        print(f"\nOpen {output_path} in your browser to view results")
    else:
        print("\n[TEST] No groups successfully consolidated")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_small_groups_only()
