"""
Embedding-based similarity grouping for cross-standard requirement harmonization.

This module uses TF-IDF vectorization and cosine similarity to identify clusters
of similar requirements across different standards. It does NOT use LLMs - just
pure vector similarity math.

Phase 0: Rule-based category classification to partition clauses before clustering.
Phase 1: TF-IDF + cosine similarity clustering within each category.
Phase 2: Cross-standard group filtering.
"""

import csv
import re
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from harmonization.models import Clause


# =============================================================================
# PHASE 0: REGULATORY CATEGORY CONSTANTS
# =============================================================================

CATEGORY_USER_DOCS = "user_documentation_and_instructions"
CATEGORY_CHARGING_STORAGE = "battery_charging_and_storage"
CATEGORY_MECHANICAL_ABUSE = "mechanical_and_abuse_tests"
CATEGORY_ELECTRICAL_SAFETY = "electrical_safety_and_insulation"
CATEGORY_ENVIRONMENTAL = "environmental_conditions"
CATEGORY_LABELING_WARNINGS = "labeling_and_warnings"
CATEGORY_DEFINITIONS_SCOPE = "definitions_and_scope"
CATEGORY_BATTERY_DESIGN = "battery_design_and_construction"
CATEGORY_PROTECTION_CIRCUITS = "protection_circuits_and_bms"
CATEGORY_MISC = "miscellaneous_or_uncategorized"

# Human-readable category titles for display/reporting
CATEGORY_TITLE_MAP = {
    CATEGORY_USER_DOCS: "User Documentation Requirements",
    CATEGORY_CHARGING_STORAGE: "Battery Charging and Storage Requirements",
    CATEGORY_MECHANICAL_ABUSE: "Mechanical and Abuse Test Requirements",
    CATEGORY_ELECTRICAL_SAFETY: "Electrical Safety and Insulation Requirements",
    CATEGORY_ENVIRONMENTAL: "Environmental Condition Requirements",
    CATEGORY_LABELING_WARNINGS: "Labeling and Warning Requirements",
    CATEGORY_DEFINITIONS_SCOPE: "Definitions and Scope",
    CATEGORY_BATTERY_DESIGN: "Battery Design and Construction Requirements",
    CATEGORY_PROTECTION_CIRCUITS: "Protection Circuit and BMS Requirements",
    CATEGORY_MISC: "Miscellaneous Safety Requirements",
}


# =============================================================================
# PHASE 0: RULE-BASED CATEGORY CLASSIFIER
# =============================================================================

def classify_clause_category(text: str) -> str:
    """
    Assign a coarse regulatory category to a clause based on its text.

    This is deterministic, fast, and does NOT use any LLMs.
    It uses keyword matching to partition clauses into regulatory categories
    before TF-IDF clustering.

    Args:
        text: The full requirement text

    Returns:
        Category string constant (one of CATEGORY_*)
    """
    t = text.lower()

    # User documentation / instructions
    if any(k in t for k in [
        "instructions for use",
        "user manual",
        "instruction manual",
        "user instructions",
        "shall be stated in the instructions",
        "information shall be provided",
        "marking in the instructions",
        "included with the product",
        "operating instructions",
        "safety instructions"
    ]):
        # More specific sub-route for charging/storage instructions
        if any(k in t for k in [
            "charging temperature",
            "charge temperature",
            "storage temperature",
            "store the battery",
            "charging procedure",
            "charger use",
            "outdoor or indoor charging"
        ]):
            return CATEGORY_CHARGING_STORAGE
        return CATEGORY_USER_DOCS

    # Labeling / warnings / symbols
    if any(k in t for k in [
        "shall be marked",
        "marking",
        "warning",
        "caution",
        "danger",
        "symbol",
        "label",
        "rating plate",
        "marking on the battery",
        "marking on the product"
    ]):
        return CATEGORY_LABELING_WARNINGS

    # Mechanical / abuse / structural
    if any(k in t for k in [
        "drop test",
        "impact test",
        "shock test",
        "vibration",
        "mechanical strength",
        "crush test",
        "compression",
        "abuse test",
        "rollover",
        "torsion",
        "bending",
        "mechanical damage"
    ]):
        return CATEGORY_MECHANICAL_ABUSE

    # Electrical safety / insulation / dielectric
    if any(k in t for k in [
        "insulation resistance",
        "dielectric strength",
        "dielectric withstand",
        "clearance",
        "creepage distance",
        "leakage current",
        "electric shock",
        "short-circuit",
        "short circuit",
        "overcurrent",
        "earth connection",
        "protective bonding"
    ]):
        return CATEGORY_ELECTRICAL_SAFETY

    # Battery design & protection
    if any(k in t for k in [
        "battery pack",
        "battery system",
        "cells and batteries",
        "cell arrangement",
        "battery enclosure",
        "venting",
        "thermal runaway",
        "protective circuit",
        "protection circuit",
        "bms",
        "battery management system",
        "overcharge protection",
        "overdischarge protection"
    ]):
        return CATEGORY_BATTERY_DESIGN

    if any(k in t for k in [
        "protection circuit",
        "protection device",
        "battery management system",
        "bms",
        "overcharge",
        "over-discharge",
        "overcurrent protection",
        "temperature cut-off",
        "fault detection"
    ]):
        return CATEGORY_PROTECTION_CIRCUITS

    # Environmental conditions
    if any(k in t for k in [
        "ambient temperature",
        "temperature range",
        "relative humidity",
        "environmental conditions",
        "ingress protection",
        "ip code",
        "dust and water",
        "altitude",
        "storage temperature",
        "transport temperature"
    ]):
        return CATEGORY_ENVIRONMENTAL

    # Definitions / scope
    if any(k in t for k in [
        "is defined as",
        "for the purposes of this standard",
        "definition",
        "this standard applies to",
        "scope",
        "does not apply to",
        "this part of",
        "normative references"
    ]):
        return CATEGORY_DEFINITIONS_SCOPE

    # Fallback
    return CATEGORY_MISC


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

            # PHASE 0: Classify clause category using rule-based classifier
            category = classify_clause_category(text_content)

            clause = Clause(
                clause_number=normalize_clause_number(clause_num),
                text=text_content.strip(),
                standard_name=standard_name.strip(),
                category=category,
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

    # Log category distribution
    category_counts = defaultdict(int)
    for clause in clauses:
        category_counts[clause.category] += 1

    print(f"[GROUPING] Loaded {len(clauses)} clauses from {csv_path}")
    print(f"[GROUPING] Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        cat_title = CATEGORY_TITLE_MAP.get(cat, cat)
        print(f"  - {cat_title}: {count} clauses")

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


def group_clauses_by_category_then_similarity(
    clauses: List[Clause],
    similarity_threshold: float = 0.3,
    embed_fn=None
) -> Tuple[List[List[int]], Dict[int, str]]:
    """
    PHASE 0 + PHASE 1: Partition clauses by category, then group within each category.

    This prevents the TF-IDF clustering from creating mega-groups on homogeneous datasets.
    Each category is clustered independently using the existing TF-IDF + graph logic.

    Args:
        clauses: List of Clause objects (with category already assigned)
        similarity_threshold: Cosine similarity threshold for grouping
        embed_fn: Optional embedding function. If None, uses TF-IDF.

    Returns:
        Tuple of (all_groups, group_to_category_map)
        - all_groups: List of groups (each group is list of indices into original clauses list)
        - group_to_category_map: Dict mapping group index to category string
    """
    print(f"\n[PHASE 0+1] Grouping {len(clauses)} clauses by category, then by similarity...")

    # Partition clauses by category
    clauses_by_category = defaultdict(list)
    clause_idx_by_category = defaultdict(list)  # Track original indices

    for idx, clause in enumerate(clauses):
        cat = clause.category or CATEGORY_MISC
        clauses_by_category[cat].append(clause)
        clause_idx_by_category[cat].append(idx)

    print(f"[PHASE 0] Partitioned into {len(clauses_by_category)} categories:")
    for cat, cat_clauses in clauses_by_category.items():
        cat_title = CATEGORY_TITLE_MAP.get(cat, cat)
        print(f"  - {cat_title}: {len(cat_clauses)} clauses")

    # Run grouping within each category
    all_groups = []
    group_to_category = {}

    for category, cat_clauses in clauses_by_category.items():
        if len(cat_clauses) < 2:
            print(f"\n[PHASE 1] Skipping category '{category}' (only {len(cat_clauses)} clause)")
            continue

        cat_title = CATEGORY_TITLE_MAP.get(category, category)
        print(f"\n[PHASE 1] Processing category: {cat_title} ({len(cat_clauses)} clauses)")

        # Compute embeddings for this category's clauses
        cat_embeddings, _ = compute_embeddings(cat_clauses, embed_fn=embed_fn)

        # Group within category using existing logic
        cat_groups = group_clauses_by_similarity(
            cat_clauses,
            cat_embeddings,
            similarity_threshold
        )

        # Map local indices back to global indices
        original_indices = clause_idx_by_category[category]
        for local_group in cat_groups:
            global_group = [original_indices[local_idx] for local_idx in local_group]
            group_idx = len(all_groups)
            all_groups.append(global_group)
            group_to_category[group_idx] = category

    print(f"\n[PHASE 1] Total groups created across all categories: {len(all_groups)}")

    return all_groups, group_to_category


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

    # PHASE 0 + PHASE 1: Group by category first, then by similarity within category
    groups, group_to_category = group_clauses_by_category_then_similarity(
        all_clauses,
        similarity_threshold=similarity_threshold,
        embed_fn=embed_fn
    )

    return all_clauses, groups
