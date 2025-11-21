"""
Embedding-based similarity grouping for cross-standard requirement harmonization.

This module uses TF-IDF vectorization and cosine similarity to identify clusters
of similar requirements across different standards. It does NOT use LLMs - just
pure vector similarity math.
"""

import csv
import re
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from harmonization.models import Clause


def normalize_clause_number(clause_num: str) -> str:
    """
    Normalize clause numbers for consistent parsing.

    For the tiny test, handles simple cases:
        "5.1.101" -> "5.1.101"
        "BB.1.2.a" -> "BB.1.2.a"
        "5.3.1(a)" -> "5.3.1.a"
        "2.1.a" -> "2.1.a"
        "  7.2.3  " -> "7.2.3"
    """
    clause_num = clause_num.strip()

    # Convert parenthesized sub-clauses: "5.3.1(a)" -> "5.3.1.a"
    if '(' in clause_num and ')' in clause_num:
        clause_num = clause_num.replace('(', '.').replace(')', '')

    return clause_num


def load_clauses_from_tagged_csv(csv_path: str) -> List[Clause]:
    """
    Load clauses from a tagged CSV file (output from validate.py).

    Expected CSV columns:
        - clause (required): Clause number like "5.1.101"
        - text (required): Full requirement text
        - Parent Section (optional): Parent section like "5.1"
        - Clause_Type (optional): Requirement/Definition/Test_Methodology/Preamble
        - Mandate_Level (optional): High/Medium/Informative
        - Safety_Flag (optional): y/n
        - Manual_Flag (optional): y/n

    The 'Standard/Reg' column is used for standard_name.
    If not present, extracts from filename (e.g., "UL_2271_2023.pdf_requirements.csv" -> "UL 2271").

    Args:
        csv_path: Path to tagged CSV file

    Returns:
        List of Clause objects
    """
    clauses = []

    # Try to extract standard name from filename as fallback
    # e.g., "UL_2271_2023.pdf_requirements.csv" -> "UL 2271"
    filename_standard = None
    if csv_path:
        filename_parts = csv_path.split('/')[-1].split('\\')[-1]  # Get filename
        match = re.search(r'([A-Z_]+)_(\d+)', filename_parts)
        if match:
            prefix = match.group(1).replace('_', ' ')
            number = match.group(2)
            filename_standard = f"{prefix} {number}"

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Support both old format (clause/text) and new format (Clause/Requirement/Description)
            clause_num = row.get('Clause/Requirement') or row.get('clause')
            text_content = row.get('Description') or row.get('text')

            # Skip empty rows
            if not clause_num or not text_content:
                continue

            # Determine standard name
            standard_name = row.get('Standard/Reg') or filename_standard or 'Unknown'

            clause = Clause(
                clause_number=normalize_clause_number(clause_num),
                text=text_content.strip(),
                standard_name=standard_name.strip(),
                parent_section=row.get('Parent Section'),
                clause_type=row.get('Clause_Type'),
                mandate_level=row.get('Mandate_Level'),
                safety_flag=row.get('Safety_Flag'),
                manual_flag=row.get('Manual_Flag'),
                requirement_scope=row.get('Requirement scope'),
                formatting_required=row.get('Formatting required?'),
                required_in_print=row.get('Required in Print?'),
                contains_image=row.get('Contains Image?'),
                safety_notice_type=row.get('Safety Notice Type'),
                comments=row.get('Comments'),
            )
            clauses.append(clause)

    print(f"[GROUPING] Loaded {len(clauses)} clauses from {csv_path}")
    return clauses


def compute_embeddings(
    clauses: List[Clause],
    embed_fn=None,
    use_cache: bool = True
) -> Tuple[np.ndarray, object]:
    """
    Compute embeddings for clause texts.

    Can use either:
    1. Real embedding function (embed_fn) - for production
    2. TF-IDF vectorization - for testing/fallback

    Args:
        clauses: List of Clause objects
        embed_fn: Optional callable that takes text and returns List[float].
                  If None, falls back to TF-IDF.
        use_cache: Whether to cache embeddings (for real embeddings only)

    Returns:
        Tuple of (embeddings matrix, vectorizer/None)
        - embeddings: np.ndarray of shape (num_clauses, embedding_dim)
        - vectorizer: TfidfVectorizer if TF-IDF was used, None otherwise
    """
    texts = [clause.text for clause in clauses]

    # REAL EMBEDDINGS MODE
    if embed_fn is not None:
        print(f"[GROUPING] Computing REAL embeddings for {len(clauses)} clauses...")
        embeddings_list = []

        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"[GROUPING]   Progress: {i + 1}/{len(texts)} clauses embedded...")

            embedding = embed_fn(text)
            embeddings_list.append(embedding)

        embeddings = np.array(embeddings_list)
        print(f"[GROUPING] Computed REAL embeddings: {embeddings.shape}")
        return embeddings, None

    # TF-IDF FALLBACK MODE
    else:
        print(f"[GROUPING] Using TF-IDF fallback (no embed_fn provided)...")

        # TF-IDF with reasonable settings for requirement text
        vectorizer = TfidfVectorizer(
            max_features=500,          # Limit vocabulary size
            stop_words='english',      # Remove common words
            ngram_range=(1, 2),        # Use unigrams and bigrams
            min_df=1,                  # Keep all terms (small dataset)
            max_df=0.95,               # Remove terms appearing in >95% of docs
        )

        embeddings = vectorizer.fit_transform(texts).toarray()

        print(f"[GROUPING] Computed TF-IDF embeddings: {embeddings.shape}")
        return embeddings, vectorizer


def group_clauses_by_similarity(
    clauses: List[Clause],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.3
) -> List[List[int]]:
    """
    Group clauses by cosine similarity using a simple threshold-based approach.

    For the tiny test, we use a threshold that results in exactly 2 groups:
    - Group 1: Safety-related requirements (battery, fire, injury, etc.)
    - Group 2: Manual/documentation requirements (instructions, warnings, markings)

    Args:
        clauses: List of Clause objects
        embeddings: TF-IDF embedding matrix
        similarity_threshold: Cosine similarity threshold (0.0 to 1.0)

    Returns:
        List of groups, where each group is a list of clause indices
    """
    n = len(clauses)

    # Compute pairwise cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    print(f"[GROUPING] Computed similarity matrix: {similarity_matrix.shape}")
    print("[GROUPING] Similarity matrix (first 10x10):")
    print(similarity_matrix[:min(10, n), :min(10, n)])

    # Build adjacency list based on similarity threshold
    print(f"[GROUPING] Building adjacency list with threshold={similarity_threshold}")
    adj = [[] for _ in range(n)]
    edge_count = 0
    for i in range(n):
        for j in range(i+1, n):
            sim_val = similarity_matrix[i][j]
            if sim_val >= similarity_threshold:
                adj[i].append(j)
                adj[j].append(i)
                edge_count += 1
                print(f"[GROUPING] Edge {edge_count}: {i} <-> {j} (sim={sim_val:.6f} >= {similarity_threshold})")
            else:
                if sim_val > 0.01:  # Only log non-zero similarities
                    print(f"[GROUPING] Skipped: {i} <-> {j} (sim={sim_val:.6f} < {similarity_threshold})")

    # Find connected components using DFS
    visited = [False] * n
    groups = []

    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            if len(component) >= 2:
                groups.append(component)

    print(f"[GROUPING] Found {len(groups)} groups (size >= 2)")

    # Filter out same-standard groups (CRITICAL: only keep cross-standard groups)
    cross_standard_groups = []
    for g in groups:
        standards = {clauses[idx].standard_name for idx in g}
        if len(standards) >= 2:
            cross_standard_groups.append(g)
        else:
            print(f"[GROUPING] Skipping same-standard group: {[clauses[idx].clause_number for idx in g]}")

    print(f"[GROUPING] Created {len(cross_standard_groups)} cross-standard groups")
    for i, group in enumerate(cross_standard_groups):
        stds = {clauses[idx].standard_name for idx in group}
        print(f"  Group {i+1}: {len(group)} clauses from {len(stds)} standards: {stds}")

    return cross_standard_groups


def load_and_group_clauses(
    csv_paths: List[str],
    similarity_threshold: float = 0.3,
    embed_fn=None
) -> Tuple[List[Clause], List[List[int]]]:
    """
    Convenience function: Load clauses from multiple CSVs and group by similarity.

    Args:
        csv_paths: List of paths to tagged CSV files
        similarity_threshold: Cosine similarity threshold for grouping
        embed_fn: Optional embedding function. If None, uses TF-IDF.

    Returns:
        Tuple of (all_clauses, groups)
        where groups is a list of lists of indices into all_clauses
    """
    # Load clauses from all CSVs
    all_clauses = []
    for csv_path in csv_paths:
        clauses = load_clauses_from_tagged_csv(csv_path)
        all_clauses.extend(clauses)

    print(f"[GROUPING] Loaded {len(all_clauses)} total clauses from {len(csv_paths)} files")

    # Compute embeddings (real or TF-IDF)
    embeddings, _ = compute_embeddings(all_clauses, embed_fn=embed_fn)

    # Group by similarity
    groups = group_clauses_by_similarity(all_clauses, embeddings, similarity_threshold)

    return all_clauses, groups
