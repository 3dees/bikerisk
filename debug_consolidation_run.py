"""Debug why consolidation run failed."""
import csv
from harmonization.models import Clause

# Load CSV
csv_path = r"C:\Users\vsjam\Downloads\all3_full_extract.csv"
print(f"[DEBUG] Loading CSV from {csv_path}")

clauses = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Map column names
        c = Clause(
            clause_number=row.get('Clause/Requirement', ''),
            text=row.get('Description', ''),
            standard_name=row.get('Standard/Reg', ''),
            category=row.get('category', 'unknown'),
            requirement_type=row.get('requirement_type', 'unknown')
        )
        clauses.append(c)

print(f"[DEBUG] Loaded {len(clauses)} clauses")

# Check category distribution
from collections import Counter
categories = Counter(c.category for c in clauses)
print(f"\n[DEBUG] Categories:")
for cat, count in categories.most_common(10):
    print(f"  {cat}: {count}")

# Check manual flags
manual_clauses = [c for c in clauses if 'user' in c.category.lower() or 'instruct' in c.category.lower() or 'manual' in c.category.lower()]
print(f"\n[DEBUG] Potential manual-related clauses: {len(manual_clauses)}")

# Try grouping simulation
print(f"\n[DEBUG] Simulating grouping with threshold 0.30...")
from harmonization.grouping import group_by_similarity_with_taxonomy
from harmonization.pipeline_unify import load_taxonomy

taxonomy = load_taxonomy()
print(f"[DEBUG] Loaded taxonomy with {len(taxonomy)} entries")

groups = group_by_similarity_with_taxonomy(clauses, threshold=0.30, taxonomy=taxonomy)
print(f"[DEBUG] Created {len(groups)} groups")

# Check group sizes
group_sizes = [len(g) for g in groups]
print(f"\n[DEBUG] Group size distribution:")
print(f"  Min: {min(group_sizes) if group_sizes else 0}")
print(f"  Max: {max(group_sizes) if group_sizes else 0}")
print(f"  Average: {sum(group_sizes)/len(group_sizes) if group_sizes else 0:.1f}")

# Check how many exceed MAX_CLAUSES_PER_GROUP = 100
large_groups = [i for i, g in enumerate(groups) if len(g) > 100]
print(f"\n[DEBUG] Groups exceeding 100 clauses: {len(large_groups)}")
if large_groups:
    print(f"  Indices: {large_groups}")
    for idx in large_groups[:5]:
        print(f"    Group {idx}: {len(groups[idx])} clauses")
