"""Run grouping on a provided multi-standard requirements CSV.

Usage: python run_real_grouping.py [optional_similarity_threshold]
"""
import sys
from harmonization.grouping import (
    load_clauses_from_tagged_csv,
    group_clauses_by_category_then_similarity,
)

CSV_PATH = r"C:\\Users\\vsjam\\Downloads\\bigger_test_consolidation - UL_2271_2023.pdf_requirements (11).csv.csv"

def main():
    threshold = 0.3
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            print(f"Invalid threshold '{sys.argv[1]}', using default 0.3")
    print(f"[RUN] Loading clauses from: {CSV_PATH}")
    clauses = load_clauses_from_tagged_csv(CSV_PATH)
    print(f"[RUN] Loaded {len(clauses)} clauses")
    groups, group_to_category = group_clauses_by_category_then_similarity(
        clauses,
        similarity_threshold=threshold,
        embed_fn=None,
    )
    print("\n[RESULT] Total groups:", len(groups))
    if not groups:
        print("[RESULT] No cross-standard groups formed at threshold", threshold)
        return
    # Summarize first 10 groups
    for i, g in enumerate(groups[:10]):
        standards = sorted({clauses[idx].standard_name for idx in g})
        clause_ids = [clauses[idx].clause_number for idx in g]
        category = group_to_category.get(i)
        req_types = sorted({clauses[idx].requirement_type for idx in g if getattr(clauses[idx], 'requirement_type', None)})
        print(f"\nGroup {i+1}: size={len(g)} standards={standards} category={category} requirement_types={req_types}")
        for idx in g[:5]:  # show up to first 5 clauses
            c = clauses[idx]
            snippet = c.text[:120].replace('\n', ' ')
            print(f"  - {c.standard_name} | {c.clause_number} | {snippet}...")
    # Distribution of group sizes
    from collections import Counter
    size_counts = Counter(len(g) for g in groups)
    print("\n[RESULT] Group size distribution:")
    for size, count in sorted(size_counts.items()):
        print(f"  size {size}: {count} groups")

if __name__ == "__main__":
    main()
