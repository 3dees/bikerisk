"""
compare_standards.py

Compare requirements across multiple standards to identify:
- Requirements appearing in all standards
- Requirements appearing in subset of standards
- Unique requirements per standard
"""

import pandas as pd
import anthropic
import httpx
import os
from rapidfuzz import fuzz
from typing import List, Dict
import json
from collections import defaultdict


def load_requirements(csv_path: str) -> List[Dict]:
    """Load and parse requirements from CSV"""
    df = pd.read_csv(csv_path)

    requirements = []
    for idx, row in df.iterrows():
        req = row.to_dict()
        req['_original_index'] = idx
        requirements.append(req)

    print(f"[COMPARE] Loaded {len(requirements)} requirements from {csv_path}")
    return requirements, df


def fuzzy_cluster_requirements(requirements: List[Dict], progress_callback=None) -> List[Dict]:
    """
    Group requirements by string similarity.
    Returns: List of clusters with metadata
    """
    print(f"[COMPARE] Starting fuzzy clustering of {len(requirements)} requirements...")
    if progress_callback:
        progress_callback("Starting fuzzy clustering...", 10)

    clusters = []
    processed = set()

    total = len(requirements)
    for i, req1 in enumerate(requirements):
        if i in processed:
            continue

        # Progress updates every 50 requirements
        if i % 50 == 0 and progress_callback:
            progress = 10 + (i / total) * 30  # 10-40% range
            progress_callback(f"Clustering requirements... ({i}/{total})", progress)

        cluster_reqs = [req1]
        processed.add(i)

        # Compare against all remaining requirements
        for j, req2 in enumerate(requirements[i+1:], start=i+1):
            if j in processed:
                continue

            similarity = fuzz.token_sort_ratio(
                str(req1.get('Description', '')),
                str(req2.get('Description', ''))
            ) / 100.0

            if similarity >= 0.70:
                cluster_reqs.append(req2)
                processed.add(j)

        # Extract unique standards in this cluster
        standards = set()
        for req in cluster_reqs:
            std = req.get('Standard/Reg', '')
            # Clean standard name (remove .pdf, page numbers, etc.)
            if std:
                # Extract base standard name
                if 'IEC' in std:
                    standards.add('IEC 62133-2')
                elif 'EN' in std or '50604' in std:
                    standards.add('SS_EN 50604')
                elif 'UL' in std or '2271' in std:
                    standards.add('UL 2271')
                else:
                    standards.add(str(std).split('_')[0])  # Fallback

        # Calculate average similarity
        if len(cluster_reqs) > 1:
            similarities = []
            for idx1 in range(len(cluster_reqs)):
                for idx2 in range(idx1 + 1, len(cluster_reqs)):
                    sim = fuzz.token_sort_ratio(
                        str(cluster_reqs[idx1].get('Description', '')),
                        str(cluster_reqs[idx2].get('Description', ''))
                    ) / 100.0
                    similarities.append(sim)
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        else:
            avg_similarity = 1.0

        # Determine match type based on similarity
        if avg_similarity >= 0.95:
            match_type = "Exact"
        elif avg_similarity >= 0.85:
            match_type = "High"
        elif avg_similarity >= 0.70:
            match_type = "Medium"
        else:
            match_type = "Unique"

        cluster = {
            'cluster_id': len(clusters),
            'requirements': cluster_reqs,
            'standards': sorted(list(standards)),
            'num_standards': len(standards),
            'avg_similarity': avg_similarity,
            'match_type': match_type
        }
        clusters.append(cluster)

    print(f"[COMPARE] Created {len(clusters)} clusters")
    if progress_callback:
        progress_callback(f"Created {len(clusters)} requirement clusters", 40)

    return clusters


def analyze_cluster_with_claude(cluster: Dict, api_key: str) -> Dict:
    """
    Send unclear clusters to Claude for semantic analysis.
    Returns: Analysis dict with equivalence determination
    """
    # Setup client
    try:
        http_client = httpx.Client(timeout=60.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception as e:
        client = anthropic.Anthropic(api_key=api_key)

    # Format requirements for prompt
    req_text = ""
    for i, req in enumerate(cluster['requirements'], 1):
        std = req.get('Standard/Reg', 'Unknown')
        clause = req.get('Clause/Requirement', 'N/A')
        desc = req.get('Description', '')
        req_text += f"{i}. [{std}] (Clause {clause}): {desc}\n\n"

    prompt = f"""You are analyzing e-bike safety standard requirements to determine if they are equivalent.

CLUSTER {cluster['cluster_id']}:
{req_text}

QUESTION: Are these requirements semantically equivalent (same regulatory intent)?

Consider:
- Do they require the same action/information?
- Are differences only in wording, or do they specify different thresholds/conditions?
- Would an engineer need to address them separately or could they satisfy all with one action?

RESPOND WITH JSON:
{{
  "are_equivalent": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "key_differences": ["List specific differences if not equivalent"],
  "consolidated_description": "If equivalent, what's the core requirement?"
}}
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        analysis = json.loads(json_str)
        return analysis

    except Exception as e:
        print(f"[COMPARE] Claude analysis failed for cluster {cluster['cluster_id']}: {e}")
        return {
            "are_equivalent": cluster['avg_similarity'] >= 0.85,
            "confidence": cluster['avg_similarity'],
            "reasoning": "Fallback to similarity score",
            "key_differences": [],
            "consolidated_description": cluster['requirements'][0].get('Description', '')
        }


def generate_comparison_report(clusters: List[Dict], original_df: pd.DataFrame, output_dir: str = "./"):
    """
    Create CSV with match groups and summary text file.
    """
    print(f"[COMPARE] Generating comparison reports...")

    # Build enhanced DataFrame with match metadata
    rows = []
    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        standards_str = ", ".join(cluster['standards'])
        num_standards = cluster['num_standards']
        similarity = cluster['avg_similarity']
        match_type = cluster['match_type']

        # Get differences summary from Claude analysis if available
        differences = cluster.get('differences_summary', '')

        for req in cluster['requirements']:
            row = req.copy()
            row['Match_Group_ID'] = cluster_id
            row['Appears_In_Standards'] = standards_str
            row['Num_Standards'] = num_standards
            row['Similarity_Score'] = round(similarity, 3)
            row['Match_Type'] = match_type
            row['Differences_Summary'] = differences
            rows.append(row)

    # Create DataFrame and save CSV
    result_df = pd.DataFrame(rows)

    # Reorder columns to put match metadata after original columns
    original_cols = list(original_df.columns)
    match_cols = ['Match_Group_ID', 'Appears_In_Standards', 'Num_Standards', 'Similarity_Score', 'Match_Type', 'Differences_Summary']
    ordered_cols = original_cols + match_cols
    result_df = result_df[ordered_cols]

    csv_path = os.path.join(output_dir, "requirements_comparison_report.csv")
    result_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[COMPARE] Saved CSV report to {csv_path}")

    # Generate summary text
    summary_path = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("REQUIREMENTS COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Group clusters by num_standards
        by_count = defaultdict(list)
        for cluster in clusters:
            by_count[cluster['num_standards']].append(cluster)

        # ALL 3 STANDARDS
        if 3 in by_count:
            f.write(f"APPEARS IN ALL 3 STANDARDS: {len(by_count[3])} groups\n")
            f.write("─" * 70 + "\n\n")

            for cluster in by_count[3][:20]:  # Show first 20
                f.write(f"Group {cluster['cluster_id']}: {cluster.get('consolidated_description', 'Cross-standard requirement')}\n")
                f.write(f"  Standards: {', '.join(cluster['standards'])}\n")
                f.write(f"  Similarity: {cluster['avg_similarity']*100:.0f}%\n\n")

                for req in cluster['requirements']:
                    std = req.get('Standard/Reg', 'Unknown')
                    clause = req.get('Clause/Requirement', 'N/A')
                    desc = req.get('Description', '')[:150]
                    f.write(f"  {std} (Clause {clause}): {desc}...\n")

                if cluster.get('key_differences'):
                    f.write(f"\n  KEY DIFFERENCES:\n")
                    for diff in cluster['key_differences']:
                        f.write(f"    - {diff}\n")

                f.write("\n" + "─" * 70 + "\n\n")

        # 2 STANDARDS
        if 2 in by_count:
            f.write(f"\nAPPEARS IN 2 STANDARDS: {len(by_count[2])} groups\n")
            f.write("─" * 70 + "\n\n")

            # Group by standard pair
            by_pair = defaultdict(list)
            for cluster in by_count[2]:
                pair = tuple(sorted(cluster['standards']))
                by_pair[pair].append(cluster)

            for pair, pair_clusters in by_pair.items():
                f.write(f"{pair[0]} + {pair[1]}: {len(pair_clusters)} groups\n")

            f.write("\n")

        # UNIQUE REQUIREMENTS
        if 1 in by_count:
            f.write(f"\nUNIQUE REQUIREMENTS: {len(by_count[1])} total\n")
            f.write("─" * 70 + "\n\n")

            # Group by standard
            by_std = defaultdict(int)
            for cluster in by_count[1]:
                std = cluster['standards'][0] if cluster['standards'] else 'Unknown'
                by_std[std] += 1

            for std, count in sorted(by_std.items()):
                f.write(f"  {std} only: {count}\n")

    print(f"[COMPARE] Saved summary report to {summary_path}")

    return {
        'csv_path': csv_path,
        'summary_path': summary_path,
        'total_requirements': len(rows),
        'cross_standard_3': len(by_count.get(3, [])),
        'cross_standard_2': len(by_count.get(2, [])),
        'unique': len(by_count.get(1, []))
    }


def compare_standards(csv_path: str, api_key: str, output_dir: str = "./", progress_callback=None):
    """
    Main function: Compare requirements across standards.

    Args:
        csv_path: Path to requirements CSV
        api_key: Anthropic API key
        output_dir: Where to save outputs
        progress_callback: Optional function(message, progress_pct) for UI updates

    Returns:
        Dict with summary statistics
    """
    # Step 1: Load requirements
    if progress_callback:
        progress_callback("Loading requirements from CSV...", 0)

    requirements, original_df = load_requirements(csv_path)

    # Step 2: Fuzzy clustering
    if progress_callback:
        progress_callback("Fuzzy clustering requirements...", 5)

    clusters = fuzzy_cluster_requirements(requirements, progress_callback)

    # Step 3: Claude analysis of unclear cases (70-95% similarity, 2+ standards)
    unclear_clusters = [
        c for c in clusters
        if 0.70 <= c['avg_similarity'] < 0.95 and c['num_standards'] >= 2
    ]

    print(f"[COMPARE] Analyzing {len(unclear_clusters)} unclear clusters with Claude...")
    if progress_callback:
        progress_callback(f"Analyzing {len(unclear_clusters)} unclear clusters with Claude...", 45)

    for i, cluster in enumerate(unclear_clusters):
        if i % 10 == 0 and progress_callback:
            progress = 45 + (i / len(unclear_clusters)) * 40  # 45-85% range
            progress_callback(f"Claude analyzing cluster {i+1}/{len(unclear_clusters)}...", progress)

        analysis = analyze_cluster_with_claude(cluster, api_key)

        # Add Claude's analysis to cluster
        cluster['consolidated_description'] = analysis.get('consolidated_description', '')
        cluster['key_differences'] = analysis.get('key_differences', [])
        cluster['differences_summary'] = '; '.join(analysis.get('key_differences', []))

        # Update match type based on Claude's equivalence determination
        if analysis.get('are_equivalent') and analysis.get('confidence', 0) >= 0.85:
            cluster['match_type'] = 'High'

    # Step 4: Generate reports
    if progress_callback:
        progress_callback("Generating comparison reports...", 90)

    result = generate_comparison_report(clusters, original_df, output_dir)

    if progress_callback:
        progress_callback(f"Complete! Found {result['cross_standard_3']} requirements in all 3 standards", 100)

    print(f"\n[COMPARE] COMPLETE!")
    print(f"  - Total requirements: {result['total_requirements']}")
    print(f"  - In all 3 standards: {result['cross_standard_3']}")
    print(f"  - In 2 standards: {result['cross_standard_2']}")
    print(f"  - Unique to 1 standard: {result['unique']}")

    return result


if __name__ == "__main__":
    # CLI interface
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compare_standards.py <csv_path> <api_key> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    api_key = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./"

    compare_standards(csv_path, api_key, output_dir)
