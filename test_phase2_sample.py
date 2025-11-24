"""Test Phase 2 consolidation with a small sample."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Create a tiny sample CSV
sample_csv = Path("test_sample_for_phase2.csv")
sample_data = """Description,Standard/Reg,Clause/Requirement,Requirement scope,Formatting required?,Required in Print?,Parent Section,Sub-section,Comments,Contains Image?,Safety Notice Type,Clause_Type,Mandate_Level,Safety_Flag,Manual_Flag
"Battery instructions shall include charging temperature range of 0-45°C.",UL 2271,5.1.101,battery,N/A,y,5.1,5.1.101,Temperature limits for charging,N,None,Requirement,Mandatory,y,y
"User manual must specify charge temperature limits between 5-40°C.",EN 50604-1,7.3.2,battery,N/A,y,7.3,7.3.2,Charging temperature specification,N,None,Requirement,Mandatory,y,y
"Charger shall be marked with voltage rating per IEC 60335-1.",IEC 60335-1,7.12,charger,Specific format required,y,7.12,7.12,Voltage marking requirement,N,None,Requirement,Mandatory,n,y"""

sample_csv.write_text(sample_data, encoding='utf-8')
print(f"[TEST] Created sample CSV: {sample_csv}")

from harmonization.pipeline_unify import call_llm_for_consolidation
from harmonization.grouping import load_and_group_clauses
from harmonization.consolidate import consolidate_groups
from harmonization.report_builder import build_html_report, save_json_report

# Load and group
csv_paths = [str(sample_csv)]
all_clauses, groups = load_and_group_clauses(csv_paths=csv_paths, similarity_threshold=0.10)  # Low threshold to force grouping

if not groups:
    print("[TEST] No groups created - clauses too dissimilar or wrong categories")
    exit(1)

print(f"\n[TEST] Created {len(groups)} groups:")
for i, group_indices in enumerate(groups):
    print(f"  Group {i}: {len(group_indices)} clauses")
    for idx in group_indices:
        cl = all_clauses[idx]
        print(f"    - [{cl.standard_name}] {cl.clause_number}: {cl.text[:60]}...")

# Consolidate with real LLM
def llm_fn(system_prompt: str, user_prompt: str) -> str:
    return call_llm_for_consolidation(system_prompt, user_prompt, use_claude=True)

print("\n[TEST] Consolidating groups with Claude API...")
req_groups = consolidate_groups(all_clauses, groups, llm_fn)

if not req_groups:
    print("[TEST] Consolidation failed - check logs above")
    exit(1)

print(f"\n[TEST] ✓ Consolidated {len(req_groups)} groups")
for rg in req_groups:
    print(f"\n  Group Title: {rg.group_title}")
    print(f"  Intent: {rg.regulatory_intent[:100]}...")
    print(f"  Consolidated: {rg.consolidated_requirement[:150]}...")
    print(f"  Differences: {len(rg.differences_across_standards)} standards")

# Save outputs
outdir = Path("outputs")
outdir.mkdir(exist_ok=True)
json_path = outdir / "test_phase2_sample.json"
html_path = outdir / "test_phase2_sample.html"

save_json_report(req_groups, str(json_path))
html_doc = build_html_report(req_groups, title="Test Phase 2 Sample Report")
html_path.write_text(html_doc, encoding='utf-8')

print(f"\n[TEST] ✓ Outputs saved:")
print(f"  JSON: {json_path}")
print(f"  HTML: {html_path}")
