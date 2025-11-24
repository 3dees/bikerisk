"""Small end-to-end grouping test exercising Phase 0.5 (requirement_type).

Creates synthetic clauses across two standards to ensure cross-standard groups
emerge after (category, requirement_type) bucketing and similarity clustering.

Run:
    python test_grouping_phase_0_5.py
"""
from harmonization.models import Clause
from harmonization.grouping import (
    group_clauses_by_category_then_similarity,
    CATEGORY_TITLE_MAP,
)

# Synthetic dataset: each text crafted to trigger specific category + requirement_type heuristics.
# Two standards: UL 2271 vs EN 15194
CLAUSES = [
    Clause(clause_number="3.1", standard_name="UL 2271", text="Definitions: For the purposes of this standard the following definitions apply."),
    Clause(clause_number="3.2", standard_name="EN 15194", text="Definitions: Terms and definitions used in this standard apply."),
    Clause(clause_number="5.1.2", standard_name="UL 2271", text="Test method: The following test method shall be carried out to verify enclosure mechanical strength."),
    Clause(clause_number="5.1.3", standard_name="EN 15194", text="Test method: The following test procedure shall be performed to verify enclosure mechanical strength."),
    Clause(clause_number="7.2.1", standard_name="UL 2271", text="Labeling: The battery shall be marked with rated voltage and manufacturer name marking label."),
    Clause(clause_number="7.2.2", standard_name="EN 15194", text="Labeling: Marking shall include rated voltage and manufacturer identification labeling symbol."),
    Clause(clause_number="8.1", standard_name="UL 2271", text="Charging instructions: Instructions for use shall include charging temperature limits and storage conditions for charging."),
    Clause(clause_number="8.2", standard_name="EN 15194", text="Charging instructions: User instructions shall specify charging temperature range and storage recommendations for charging."),
    Clause(clause_number="9.4", standard_name="UL 2271", text="Warning: Do not disassemble. Risk of fire or electric shock hazard warning."),
    Clause(clause_number="9.5", standard_name="EN 15194", text="Warning: Risk of electric shock hazard. Do not open the battery enclosure warning."),
]


def main():
    print("=" * 80)
    print("PHASE 0.5 GROUPING TEST (Synthetic Clauses)")
    print("=" * 80)
    print(f"Total clauses: {len(CLAUSES)}\n")

    groups, group_to_category = group_clauses_by_category_then_similarity(
        CLAUSES,
        similarity_threshold=0.05,  # Low threshold to increase grouping connectivity for synthetic text
        embed_fn=None  # TF-IDF fallback
    )

    if not groups:
        print("No groups formed. Check similarity_threshold or diversity across standards.")
        return

    print(f"\nCross-standard groups formed: {len(groups)}\n")

    for i, grp in enumerate(groups):
        cat = group_to_category.get(i, "unknown")
        cat_title = CATEGORY_TITLE_MAP.get(cat, cat)
        # Collect requirement types in this group
        req_types = sorted({CLAUSES[idx].requirement_type for idx in grp})
        standards = sorted({CLAUSES[idx].standard_name for idx in grp})
        print(f"Group {i+1}: {len(grp)} clauses | Category={cat_title} | requirement_types={req_types} | standards={standards}")
        for idx in grp:
            c = CLAUSES[idx]
            preview = c.text[:70] + ("..." if len(c.text) > 70 else "")
            print(f"  - [{c.standard_name}] {c.clause_number} | req_type={c.requirement_type} | {preview}")
        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
