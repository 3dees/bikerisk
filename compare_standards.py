"""
compare_standards.py

Compare requirements across multiple standards using the same
similarity matching as extraction (SequenceMatcher).

This tool:
1. Clusters requirements by text similarity
2. Identifies cross-standard overlaps
3. Extracts key differences (temperatures, measurements, etc.)
4. Generates CSV + human-readable report
"""

from difflib import SequenceMatcher
from collections import defaultdict
from typing import List, Dict
import pandas as pd
import re
import os


def load_requirements(csv_path: str) -> List[Dict]:
    """Load requirements from CSV, preserving all columns"""
    df = pd.read_csv(csv_path)
    print(f"[COMPARE] Loaded {len(df)} requirements from {csv_path}")
    return df.to_dict('records'), df


def fuzzy_cluster_requirements(requirements: List[Dict], similarity_threshold: float = 0.70, progress_callback=None) -> List[Dict]:
    """
    Group requirements by text similarity using SequenceMatcher.
    Returns: List of clusters, each cluster contains similar requirements.

    Args:
        requirements: List of dicts with 'Description', 'Standard/Reg', 'Clause/Requirement'
        similarity_threshold: Minimum similarity to group (default 0.70)
        progress_callback: Optional function(message, progress_pct) for UI updates
    """
    clusters = []
    processed = set()
    total = len(requirements)

    print(f"[COMPARE] Clustering {total} requirements with {similarity_threshold:.0%} threshold...")
    if progress_callback:
        progress_callback(f"Clustering {total} requirements...", 10)

    for i, req1 in enumerate(requirements):
        if i in processed:
            continue

        # Progress updates every 50 requirements
        if i % 50 == 0 and progress_callback:
            progress = 10 + (i / total) * 70  # 10-80% range
            progress_callback(f"Clustering... ({i}/{total} processed)", progress)

        cluster = {
            'cluster_id': len(clusters),
            'requirements': [req1],
            'standards': set(),
            'similarities': []
        }

        # Extract and clean standard name
        std1 = req1.get('Standard/Reg', '')
        if 'IEC' in std1:
            cluster['standards'].add('IEC 62133-2')
        elif 'EN' in std1 or '50604' in std1:
            cluster['standards'].add('SS_EN 50604')
        elif 'UL' in std1 or '2271' in std1:
            cluster['standards'].add('UL 2271')
        else:
            cluster['standards'].add(str(std1).split('_')[0])

        processed.add(i)

        # Find similar requirements
        desc1 = str(req1.get('Description', '')).strip().lower()

        for j, req2 in enumerate(requirements[i+1:], start=i+1):
            if j in processed:
                continue

            desc2 = str(req2.get('Description', '')).strip().lower()

            # Calculate similarity using SequenceMatcher (same as extraction)
            similarity = SequenceMatcher(None, desc1, desc2).ratio()

            if similarity >= similarity_threshold:
                cluster['requirements'].append(req2)
                cluster['similarities'].append(similarity)

                # Extract and clean standard name
                std2 = req2.get('Standard/Reg', '')
                if 'IEC' in std2:
                    cluster['standards'].add('IEC 62133-2')
                elif 'EN' in std2 or '50604' in std2:
                    cluster['standards'].add('SS_EN 50604')
                elif 'UL' in std2 or '2271' in std2:
                    cluster['standards'].add('UL 2271')
                else:
                    cluster['standards'].add(str(std2).split('_')[0])

                processed.add(j)

        # Calculate average similarity for cluster
        if cluster['similarities']:
            cluster['avg_similarity'] = sum(cluster['similarities']) / len(cluster['similarities'])
        else:
            cluster['avg_similarity'] = 1.0  # Single requirement = perfect match with itself

        # Determine match type
        if cluster['avg_similarity'] >= 0.95:
            cluster['match_type'] = 'Exact'
        elif cluster['avg_similarity'] >= 0.85:
            cluster['match_type'] = 'High'
        elif cluster['avg_similarity'] >= 0.70:
            cluster['match_type'] = 'Medium'
        else:
            cluster['match_type'] = 'Unique'

        clusters.append(cluster)

    print(f"[COMPARE] Created {len(clusters)} clusters")
    if progress_callback:
        progress_callback(f"Created {len(clusters)} clusters", 80)

    return clusters


def extract_key_differences(cluster: Dict) -> str:
    """
    Extract specific differences between requirements in a cluster.
    Focus on: temperatures, measurements, format requirements, warnings.
    """
    differences = []

    for req in cluster['requirements']:
        text = str(req.get('Description', ''))
        std_raw = req.get('Standard/Reg', '')

        # Clean standard name for display
        if 'IEC' in std_raw:
            std = 'IEC'
        elif 'EN' in std_raw or '50604' in std_raw:
            std = 'EN'
        elif 'UL' in std_raw or '2271' in std_raw:
            std = 'UL'
        else:
            std = str(std_raw).split('_')[0][:10]

        # Temperature specifics
        temp_match = re.findall(r'[-\d]+\s*°[CF]', text)
        if temp_match:
            differences.append(f"{std}: {', '.join(temp_match)}")

        # Measurements
        measure_match = re.findall(r'\d+\.?\d*\s*(mm|cm|m|inch|")', text)
        if measure_match:
            differences.append(f"{std}: {', '.join(set(measure_match))}")

        # Format requirements
        if 'paper' in text.lower() and 'paper' not in ' '.join(differences).lower():
            differences.append(f"{std}: Paper format")
        if 'digital' in text.lower() and 'digital' not in ' '.join(differences).lower():
            differences.append(f"{std}: Digital format")

        # Voltage/current specifics
        voltage_match = re.findall(r'\d+\.?\d*\s*[Vv](?:olts?)?', text)
        if voltage_match:
            differences.append(f"{std}: {', '.join(set(voltage_match))}")

        # Time durations
        time_match = re.findall(r'\d+\.?\d*\s*(seconds?|minutes?|hours?|days?|s|min|h)', text, re.IGNORECASE)
        if time_match:
            differences.append(f"{std}: {', '.join(set(time_match))}")

    return '; '.join(differences) if differences else 'No specific differences detected'


def generate_summary_text(clusters: List[Dict]) -> str:
    """Create human-readable summary"""
    # Group by number of standards
    all_3 = [c for c in clusters if len(c['standards']) == 3]
    two_std = [c for c in clusters if len(c['standards']) == 2]
    unique = [c for c in clusters if len(c['standards']) == 1]

    summary = "REQUIREMENTS COMPARISON SUMMARY\n"
    summary += "=" * 70 + "\n\n"

    # Statistics overview
    summary += f"Total Clusters: {len(clusters)}\n"
    summary += f"Cross-Standard Groups (2+ standards): {len(all_3) + len(two_std)}\n"
    summary += f"Unique to One Standard: {len(unique)}\n\n"

    # All 3 standards
    summary += f"APPEARS IN ALL 3 STANDARDS: {len(all_3)} groups\n"
    summary += "─" * 70 + "\n"
    for cluster in all_3[:20]:  # Show first 20
        summary += f"\nGROUP {cluster['cluster_id']}: (Similarity: {cluster['avg_similarity']:.0%})\n"
        for req in cluster['requirements']:
            std_raw = req.get('Standard/Reg', 'Unknown')
            # Clean standard name
            if 'IEC' in std_raw:
                std = 'IEC 62133-2'
            elif 'EN' in std_raw or '50604' in std_raw:
                std = 'SS_EN 50604'
            elif 'UL' in std_raw or '2271' in std_raw:
                std = 'UL 2271'
            else:
                std = str(std_raw).split('_')[0]

            clause = req.get('Clause/Requirement', 'N/A')
            text = str(req.get('Description', ''))[:80]
            summary += f"  - {std} (Clause {clause}): {text}...\n"

        diff = extract_key_differences(cluster)
        if diff != 'No specific differences detected':
            summary += f"  Key Differences: {diff}\n"

    # 2 standards
    summary += f"\n\nAPPEARS IN 2 STANDARDS: {len(two_std)} groups\n"
    summary += "─" * 70 + "\n"

    # Group by standard pairs
    pairs = defaultdict(int)
    for cluster in two_std:
        pair = tuple(sorted(cluster['standards']))
        pairs[pair] += 1

    for pair, count in sorted(pairs.items()):
        summary += f"  {' + '.join(pair)}: {count} groups\n"

    # Unique
    summary += f"\n\nUNIQUE REQUIREMENTS: {len(unique)} total\n"
    summary += "─" * 70 + "\n"

    unique_by_std = defaultdict(int)
    for cluster in unique:
        std = list(cluster['standards'])[0]
        unique_by_std[std] += len(cluster['requirements'])

    for std, count in sorted(unique_by_std.items()):
        summary += f"  {std}: {count} requirements\n"

    return summary


def generate_comparison_report(clusters: List[Dict], original_df: pd.DataFrame, output_dir: str = "./", progress_callback=None):
    """Generate both CSV and text reports"""
    print("[COMPARE] Generating reports...")
    if progress_callback:
        progress_callback("Generating comparison reports...", 85)

    # Build CSV
    csv_rows = []
    for cluster in clusters:
        standards_str = ', '.join(sorted(cluster['standards']))
        key_diff = extract_key_differences(cluster)

        for req in cluster['requirements']:
            row = {
                **req,  # All original columns
                'Match_Group_ID': cluster['cluster_id'],
                'Appears_In_Standards': standards_str,
                'Num_Standards': len(cluster['standards']),
                'Similarity_Score': round(cluster['avg_similarity'], 3),
                'Match_Type': cluster['match_type'],
                'Key_Differences': key_diff
            }
            csv_rows.append(row)

    df = pd.DataFrame(csv_rows)

    # Reorder columns to put match metadata after original columns
    original_cols = list(original_df.columns)
    match_cols = ['Match_Group_ID', 'Appears_In_Standards', 'Num_Standards', 'Similarity_Score', 'Match_Type', 'Key_Differences']
    ordered_cols = original_cols + match_cols
    df = df[ordered_cols]

    output_csv = os.path.join(output_dir, "requirements_comparison_report.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"[COMPARE] Saved CSV report: {output_csv}")

    # Build summary text
    summary = generate_summary_text(clusters)
    output_summary = os.path.join(output_dir, "comparison_summary.txt")
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"[COMPARE] Saved summary report: {output_summary}")

    if progress_callback:
        progress_callback("Reports generated successfully", 95)

    return output_csv, output_summary


def compare_standards(csv_path: str, output_dir: str = "./", progress_callback=None) -> Dict:
    """
    Main function: Compare requirements across standards.

    Args:
        csv_path: Path to requirements CSV
        output_dir: Where to save outputs
        progress_callback: Optional function(message, progress_pct) for UI updates

    Returns:
        Dict with summary statistics and file paths
    """
    if progress_callback:
        progress_callback("Loading requirements from CSV...", 0)

    print("[COMPARE] Loading requirements...")
    requirements, original_df = load_requirements(csv_path)

    print(f"[COMPARE] Clustering {len(requirements)} requirements...")
    if progress_callback:
        progress_callback(f"Clustering {len(requirements)} requirements...", 5)

    clusters = fuzzy_cluster_requirements(requirements, similarity_threshold=0.70, progress_callback=progress_callback)

    print(f"[COMPARE] Found {len(clusters)} clusters")

    # Generate reports
    output_csv, output_summary = generate_comparison_report(clusters, original_df, output_dir, progress_callback)

    # Calculate statistics
    all_3 = len([c for c in clusters if len(c['standards']) == 3])
    two_std = len([c for c in clusters if len(c['standards']) == 2])
    unique = len([c for c in clusters if len(c['standards']) == 1])

    stats = {
        'total_requirements': len(requirements),
        'total_clusters': len(clusters),
        'cross_standard_3': all_3,
        'cross_standard_2': two_std,
        'unique': unique,
        'csv_path': output_csv,
        'summary_path': output_summary
    }

    print(f"\n[COMPARE] Results:")
    print(f"  - Total requirements: {stats['total_requirements']}")
    print(f"  - Total clusters: {stats['total_clusters']}")
    print(f"  - In all 3 standards: {stats['cross_standard_3']}")
    print(f"  - In 2 standards: {stats['cross_standard_2']}")
    print(f"  - Unique to 1 standard: {stats['unique']}")

    if progress_callback:
        progress_callback(f"Complete! Found {all_3} requirements in all 3 standards", 100)

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compare_standards.py <csv_path> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./"

    compare_standards(csv_path, output_dir)
