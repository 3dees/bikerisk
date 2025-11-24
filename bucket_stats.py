import sys, csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from harmonization.grouping import (
    load_clauses_from_tagged_csv,
    enrich_clauses_with_metadata,
    CATEGORY_MISC,
)

TARGET_CATEGORY = 'miscellaneous_or_uncategorized'
TARGET_REQ_TYPE = 'misc'


def print_similarity_stats(similarity_matrix: np.ndarray, bucket_name: str):
    n = similarity_matrix.shape[0]
    if n < 2:
        print(f"[STATS] Bucket {bucket_name} has <2 clauses; skipping")
        return
    tri_indices = np.triu_indices(n, k=1)
    sims = similarity_matrix[tri_indices]
    if sims.size == 0:
        print(f"[STATS] No pairwise sims for {bucket_name}")
        return
    percentiles = {
        'median': np.percentile(sims, 50),
        'p75': np.percentile(sims, 75),
        'p90': np.percentile(sims, 90),
        'p95': np.percentile(sims, 95),
        'p99': np.percentile(sims, 99),
    }
    edge_counts = {
        '>=0.30': int(np.sum(sims >= 0.30)),
        '>=0.35': int(np.sum(sims >= 0.35)),
        '>=0.40': int(np.sum(sims >= 0.40)),
        '>=0.45': int(np.sum(sims >= 0.45)),
        '>=0.50': int(np.sum(sims >= 0.50)),
    }
    bins = np.linspace(0.0, 1.0, 21)
    hist, _ = np.histogram(sims, bins=bins)
    bucket_ranges = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    print(f"[STATS] Bucket: {bucket_name} | size={n} | pairs={sims.size}")
    print("[STATS] Percentiles:")
    for k, v in percentiles.items():
        print(f"  - {k}: {v:.4f}")
    print("[STATS] Edge counts:")
    for k, v in edge_counts.items():
        print(f"  - {k}: {v}")
    print("[STATS] Histogram (range => count):")
    for rng, count in zip(bucket_ranges, hist):
        print(f"  - {rng}: {count}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python bucket_stats.py <csv_path>")
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    clauses = load_clauses_from_tagged_csv(str(csv_path))
    clauses = enrich_clauses_with_metadata(clauses)
    bucket = [c for c in clauses if (c.category or CATEGORY_MISC) == TARGET_CATEGORY and (c.requirement_type or 'misc') == TARGET_REQ_TYPE]
    print(f"[RUN] Total clauses loaded: {len(clauses)}")
    print(f"[RUN] Target bucket '{TARGET_CATEGORY}/{TARGET_REQ_TYPE}' size: {len(bucket)}")
    texts = [c.text for c in bucket]
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2), min_df=1, max_df=0.95)
    embeddings = vectorizer.fit_transform(texts).toarray()
    sim_matrix = cosine_similarity(embeddings)
    print_similarity_stats(sim_matrix, f"{TARGET_CATEGORY}/{TARGET_REQ_TYPE}")

if __name__ == '__main__':
    main()
