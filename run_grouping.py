"""Generic grouping runner for one tagged requirements CSV.

Usage:
    python run_grouping.py <csv_path> [similarity_threshold]

Example:
    python run_grouping.py "C:\\Users\\vsjam\\Downloads\\3standards - 3standards.csv" 0.25
"""
import sys
from pathlib import Path
from harmonization.grouping import (
    load_clauses_from_tagged_csv,
    group_clauses_by_category_then_similarity,
)


def main():
    if len(sys.argv) < 2:
        print("ERROR: csv_path argument required")
        sys.exit(1)
    csv_path = sys.argv[1]
    threshold = 0.3
    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            print(f"Invalid threshold '{sys.argv[2]}', using default {threshold}")
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print(f"[RUN] CSV: {csv_path}")
    print(f"[RUN] Similarity threshold: {threshold}")

    clauses = load_clauses_from_tagged_csv(csv_path)
    print(f"[RUN] Loaded {len(clauses)} clauses")

    groups, group_to_category = group_clauses_by_category_then_similarity(
        clauses,
        similarity_threshold=threshold,
        embed_fn=None,
    )

    print("\n[RESULT] Total cross-standard groups:", len(groups))
    if not groups:
        print("[RESULT] No cross-standard groups formed at threshold", threshold)
        return

    # Summarize
    from collections import Counter
    size_counts = Counter(len(g) for g in groups)
    print("[RESULT] Group size distribution:")
    for size, count in sorted(size_counts.items()):
        print(f"  size {size}: {count} groups")

    # Show first 15 groups brief
    for i, g in enumerate(groups[:15]):
        standards = sorted({clauses[idx].standard_name for idx in g})
        category = group_to_category.get(i)
        req_types = sorted({clauses[idx].requirement_type for idx in g if getattr(clauses[idx], 'requirement_type', None)})
        print(f"\nGroup {i+1}: size={len(g)} standards={standards} category={category} requirement_types={req_types}")
        for idx in g[:6]:  # show a few clauses
            c = clauses[idx]
            snippet = c.text[:140].replace('\n', ' ')
            print(f"  - {c.standard_name} | {c.clause_number} | {snippet}...")

if __name__ == "__main__":
    main()
