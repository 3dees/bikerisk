"""Run Phase 2 consolidation on tagged requirements CSVs and save outputs.

Usage (Windows PowerShell):
    python run_phase2_consolidation.py -i path\to\A.csv path\to\B.csv -o outputs -t 0.30

Notes:
- Requires ANTHROPIC_API_KEY set in the environment (Claude Sonnet 4.5 required).
- Uses TF-IDF embeddings internally for grouping unless embed_fn is provided.
- Dynamic thresholds inside grouping are respected; provide a base threshold via -t.
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from harmonization.pipeline_unify import (
    call_llm_for_consolidation,
)
from harmonization.grouping import load_and_group_clauses
from harmonization.consolidate import consolidate_groups
from harmonization.report_builder import build_html_report, save_json_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 consolidation runner")
    p.add_argument(
        "-i", "--inputs", nargs="+", required=True,
        help="One or more tagged requirements CSV files"
    )
    p.add_argument(
        "-o", "--outdir", default="outputs",
        help="Output directory (default: outputs)"
    )
    p.add_argument(
        "-t", "--threshold", type=float, default=0.30,
        help="Base similarity threshold for small buckets (default: 0.30)"
    )
    p.add_argument(
        "--html", default="harmonization_report.html",
        help="HTML report filename (default: harmonization_report.html)"
    )
    p.add_argument(
        "--json", default="harmonization_groups.json",
        help="JSON output filename (default: harmonization_groups.json)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate inputs
    csv_paths = [str(Path(p)) for p in args.inputs]
    for p in csv_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_threshold = float(args.threshold)
    print("=" * 80)
    print("PHASE 2 CONSOLIDATION - RUNNER")
    print("=" * 80)
    print(f"Inputs: {csv_paths}")
    print(f"Output dir: {outdir}")
    print(f"Base similarity threshold: {base_threshold:.2f}")

    # Step 1: Load and group clauses (TF-IDF embeddings, dynamic thresholds inside grouping)
    all_clauses, groups = load_and_group_clauses(csv_paths=csv_paths, similarity_threshold=base_threshold)
    if not groups:
        print("[PHASE2] No cross-standard groups found. Nothing to consolidate.")
        return
    print(f"[PHASE2] Cross-standard groups: {len(groups)}")

    # Step 2: Build Claude Sonnet 4.5 consolidation function
    def llm_fn(system_prompt: str, user_prompt: str) -> str:
        return call_llm_for_consolidation(system_prompt, user_prompt, use_claude=True)

    # Step 3: Consolidate groups via LLM
    req_groups = consolidate_groups(all_clauses, groups, llm_fn)
    if not req_groups:
        print("[PHASE2] No groups consolidated (all failed or skipped).")
        return

    # Step 4: Save JSON + HTML outputs
    json_path = outdir / args.json
    html_path = outdir / args.html

    save_json_report(req_groups, str(json_path))
    html_doc = build_html_report(req_groups, title="Cross-Standard Harmonization Report")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print("=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"Consolidated groups: {len(req_groups)}")
    print(f"JSON saved: {json_path}")
    print(f"HTML saved: {html_path}")


if __name__ == "__main__":
    main()
