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
# PHASE 0.5: REQUIREMENT TYPE CONSTANTS
# =============================================================================

REQUIREMENT_TYPE_DEFINITIONS = "definitions"
REQUIREMENT_TYPE_SCOPE = "scope"
REQUIREMENT_TYPE_TEST_METHOD = "test_method"
REQUIREMENT_TYPE_TEST_ACCEPTANCE = "test_acceptance"
REQUIREMENT_TYPE_BMS_BEHAVIOR = "bms_behavior"
REQUIREMENT_TYPE_DESIGN_CONSTRUCTION = "design_construction"
REQUIREMENT_TYPE_LABELING = "labeling"
REQUIREMENT_TYPE_WARNINGS = "warnings"
REQUIREMENT_TYPE_USER_INSTRUCTIONS = "user_instructions"
REQUIREMENT_TYPE_CHARGING_STORAGE = "charging_storage"
REQUIREMENT_TYPE_ENVIRONMENTAL = "environmental"
REQUIREMENT_TYPE_DISPOSAL_RECYCLING = "disposal_recycling"
REQUIREMENT_TYPE_PERFORMANCE_LIMITS = "performance_limits"
REQUIREMENT_TYPE_MISC = "misc"


"""Embedding-based similarity grouping for cross-standard requirement harmonization.

This module uses TF-IDF vectorization and cosine similarity to identify clusters
of similar requirements across different standards. It does NOT use LLMs - just
pure vector similarity math.

Phase 0: Rule-based category classification to partition clauses before clustering.
Phase 0.5: Requirement type enrichment (requirement_type).
Phase 1: TF-IDF + cosine similarity clustering within each (category, requirement_type) bucket.
Phase 2: Cross-standard group filtering.
"""

import csv
import re
import os
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from harmonization.models import Clause

# Quiet mode (set env GROUPING_QUIET=1 to suppress per-edge similarity logging)
QUIET = os.environ.get("GROUPING_QUIET", "0") == "1"

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
# PHASE 0.5: REQUIREMENT TYPE CONSTANTS
# =============================================================================

REQUIREMENT_TYPE_DEFINITIONS = "definitions"
REQUIREMENT_TYPE_SCOPE = "scope"
REQUIREMENT_TYPE_TEST_METHOD = "test_method"
REQUIREMENT_TYPE_TEST_ACCEPTANCE = "test_acceptance"
REQUIREMENT_TYPE_BMS_BEHAVIOR = "bms_behavior"
REQUIREMENT_TYPE_DESIGN_CONSTRUCTION = "design_construction"
REQUIREMENT_TYPE_LABELING = "labeling"
REQUIREMENT_TYPE_WARNINGS = "warnings"
REQUIREMENT_TYPE_USER_INSTRUCTIONS = "user_instructions"
REQUIREMENT_TYPE_CHARGING_STORAGE = "charging_storage"
REQUIREMENT_TYPE_ENVIRONMENTAL = "environmental"
REQUIREMENT_TYPE_DISPOSAL_RECYCLING = "disposal_recycling"
REQUIREMENT_TYPE_PERFORMANCE_LIMITS = "performance_limits"
REQUIREMENT_TYPE_MISC = "misc"

# =============================================================================
# PHASE 0: RULE-BASED CATEGORY CLASSIFIER
# =============================================================================

def classify_clause_category(text: str) -> str:
    """Assign a coarse regulatory category to a clause based on its text."""
    t = text.lower()

    if any(k in t for k in [
        "instructions for use","user manual","instruction manual","user instructions",
        "shall be stated in the instructions","information shall be provided","marking in the instructions",
        "included with the product","operating instructions","safety instructions"
    ]):
        if any(k in t for k in [
            "charging temperature","charge temperature","storage temperature","store the battery",
            "charging procedure","charger use","outdoor or indoor charging"
        ]):
            return CATEGORY_CHARGING_STORAGE
        return CATEGORY_USER_DOCS

    if any(k in t for k in [
        "shall be marked","marking","warning","caution","danger","symbol","label",
        "rating plate","marking on the battery","marking on the product"
    ]):
        return CATEGORY_LABELING_WARNINGS

    if any(k in t for k in [
        "drop test","impact test","shock test","vibration","mechanical strength","crush test",
        "compression","abuse test","rollover","torsion","bending","mechanical damage"
    ]):
        return CATEGORY_MECHANICAL_ABUSE

    if any(k in t for k in [
        "insulation resistance","dielectric strength","dielectric withstand","clearance",
        "creepage distance","leakage current","electric shock","short-circuit","short circuit",
        "overcurrent","earth connection","protective bonding"
    ]):
        return CATEGORY_ELECTRICAL_SAFETY

    if any(k in t for k in [
        "battery pack","battery system","cells and batteries","cell arrangement","battery enclosure",
        "venting","thermal runaway","protective circuit","protection circuit","bms",
        "battery management system","overcharge protection","overdischarge protection"
    ]):
        return CATEGORY_BATTERY_DESIGN

    if any(k in t for k in [
        "protection circuit","protection device","battery management system","bms","overcharge",
        "over-discharge","overcurrent protection","temperature cut-off","fault detection"
    ]):
        return CATEGORY_PROTECTION_CIRCUITS

    if any(k in t for k in [
        "ambient temperature","temperature range","relative humidity","environmental conditions",
        "ingress protection","ip code","dust and water","altitude","storage temperature",
        "transport temperature"
    ]):
        return CATEGORY_ENVIRONMENTAL

    if any(k in t for k in [
        "is defined as","for the purposes of this standard","definition","this standard applies to",
        "scope","does not apply to","this part of","normative references"
    ]):
        return CATEGORY_DEFINITIONS_SCOPE

    return CATEGORY_MISC

# =============================================================================
# PHASE 0.5: REQUIREMENT TYPE CLASSIFIER
# =============================================================================

def classify_requirement_type(clause: Clause) -> str:
    text = clause.text.lower()
    clause_num = (clause.clause_number or "").lower()

    if "definition" in text or clause_num.startswith("3.") or "terms and definitions" in text:
        return REQUIREMENT_TYPE_DEFINITIONS
    if "scope" in text and ("applies to" in text or "this standard" in text):
        return REQUIREMENT_TYPE_SCOPE
    if any(k in text for k in [
        "test method","test procedure","the following test shall be carried out",
        "the following test shall be conducted","test is carried out","test is performed"
    ]):
        return REQUIREMENT_TYPE_TEST_METHOD
    if any(k in text for k in [
        "the battery shall not","no rupture shall occur","no fire shall occur","no explosion shall occur",
        "shall remain within the specified limits","criteria are fulfilled","compliance is checked by"
    ]):
        return REQUIREMENT_TYPE_TEST_ACCEPTANCE
    if any(k in text for k in [
        "battery management system","bms","protection circuit","overcharge protection",
        "overdischarge protection","overcurrent protection","cell balancing","cut-off device"
    ]):
        return REQUIREMENT_TYPE_BMS_BEHAVIOR
    if any(k in text for k in [
        "shall be constructed","shall be designed","mechanical construction","enclosure shall",
        "creepage distance","clearance","insulation","mechanical strength","fixing means"
    ]):
        return REQUIREMENT_TYPE_DESIGN_CONSTRUCTION
    if any(k in text for k in [
        "shall be marked","marking","label","labels","rating plate","symbol","symbol for"
    ]):
        return REQUIREMENT_TYPE_LABELING
    if any(k in text for k in [
        "warning:","warnings shall","caution:","do not disassemble","risk of","hazard"
    ]):
        return REQUIREMENT_TYPE_WARNINGS
    if any(k in text for k in [
        "instructions for use","instructions shall","user instructions","the instructions shall",
        "instruction manual","accompanying documents"
    ]):
        if any(k in text for k in ["charge","charging","storage","store","transport"]):
            return REQUIREMENT_TYPE_CHARGING_STORAGE
        return REQUIREMENT_TYPE_USER_INSTRUCTIONS
    if any(k in text for k in [
        "charging","charge the battery","recharge","charging temperature","storage temperature",
        "store the battery","transport","shipping conditions"
    ]):
        return REQUIREMENT_TYPE_CHARGING_STORAGE
    if any(k in text for k in [
        "ambient temperature","humidity","vibration","shock","altitude","environmental conditions"
    ]):
        return REQUIREMENT_TYPE_ENVIRONMENTAL
    if any(k in text for k in [
        "disposal","dispose of","recycle","recycling","end-of-life","waste electrical and electronic equipment","weee"
    ]):
        return REQUIREMENT_TYPE_DISPOSAL_RECYCLING
    if any(k in text for k in [
        "rated voltage","rated capacity","current rating","maximum charge current","maximum discharge current","limits shall not be exceeded"
    ]):
        return REQUIREMENT_TYPE_PERFORMANCE_LIMITS
    return REQUIREMENT_TYPE_MISC

def enrich_clauses_with_metadata(clauses: List[Clause]) -> List[Clause]:
    print(f"[PHASE 0+0.5] Enriching {len(clauses)} clauses with metadata...")
    for clause in clauses:
        if not clause.category:
            clause.category = classify_clause_category(clause.text)
        if not clause.requirement_type:
            clause.requirement_type = classify_requirement_type(clause)
    bucket_counts = defaultdict(int)
    for clause in clauses:
        key = (clause.category or CATEGORY_MISC, clause.requirement_type or REQUIREMENT_TYPE_MISC)
        bucket_counts[key] += 1
    print("[PHASE 0+0.5] Metadata enrichment complete")
    print("[PHASE 0+0.5] Distribution by (category, requirement_type):")
    for (cat, req_type), count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        cat_title = CATEGORY_TITLE_MAP.get(cat, cat)
        print(f"  - {cat_title} / {req_type}: {count} clauses")
    return clauses

def normalize_clause_number(clause_num: str) -> str:
    clause_num = clause_num.strip()
    if '(' in clause_num and ')' in clause_num:
        clause_num = clause_num.replace('(', '.').replace(')', '')
    return clause_num

def load_clauses_from_tagged_csv(csv_path: str) -> List[Clause]:
    clauses = []
    filename_standard = None
    if csv_path:
        filename_parts = csv_path.split('/')[-1].split('\\')[-1]
        match = re.search(r'([A-Z_]+)_(\d+)', filename_parts)
        if match:
            prefix = match.group(1).replace('_', ' ')
            number = match.group(2)
            filename_standard = f"{prefix} {number}"

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clause_num = row.get('Clause/Requirement') or row.get('clause')
            text_content = row.get('Description') or row.get('text')
            if not clause_num or not text_content:
                continue
            standard_name = row.get('Standard/Reg') or filename_standard or 'Unknown'
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

    category_counts = defaultdict(int)
    for clause in clauses:
        category_counts[clause.category] += 1
    print(f"[GROUPING] Loaded {len(clauses)} clauses from {csv_path}")
    print("[GROUPING] Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        cat_title = CATEGORY_TITLE_MAP.get(cat, cat)
        print(f"  - {cat_title}: {count} clauses")
    return clauses

def compute_embeddings(clauses: List[Clause], embed_fn=None, use_cache: bool = True) -> Tuple[np.ndarray, object]:
    texts = [clause.text for clause in clauses]
    if embed_fn is not None:
        print(f"[GROUPING] Computing REAL embeddings for {len(clauses)} clauses...")
        embeddings_list = []
        for i, text in enumerate(texts):
            embedding = embed_fn(text)
            embeddings_list.append(embedding)
        embeddings = np.array(embeddings_list)
        print(f"[GROUPING] Computed REAL embeddings: {embeddings.shape}")
        return embeddings, None
    print(f"[GROUPING] Using TF-IDF fallback (no embed_fn provided)...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2), min_df=1, max_df=0.95)
    embeddings = vectorizer.fit_transform(texts).toarray()
    print(f"[GROUPING] Computed TF-IDF embeddings: {embeddings.shape}")
    return embeddings, vectorizer

def group_clauses_by_similarity(clauses: List[Clause], embeddings: np.ndarray, similarity_threshold: float = 0.3) -> List[List[int]]:
    n = len(clauses)
    similarity_matrix = cosine_similarity(embeddings)
    if not QUIET:
        print(f"[GROUPING] Computed similarity matrix: {similarity_matrix.shape}")
        print("[GROUPING] Similarity matrix (first 10x10):")
        print(similarity_matrix[:min(10,n), :min(10,n)])
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
                if not QUIET:
                    print(f"[GROUPING] Edge {edge_count}: {i} <-> {j} (sim={sim_val:.6f} >= {similarity_threshold})")
            else:
                if sim_val > 0.01 and not QUIET:
                    print(f"[GROUPING] Skipped: {i} <-> {j} (sim={sim_val:.6f} < {similarity_threshold})")
    visited = [False]*n
    groups: List[List[int]] = []
    def dfs(node, comp):
        visited[node]=True
        comp.append(node)
        for nb in adj[node]:
            if not visited[nb]:
                dfs(nb, comp)
    for i in range(n):
        if not visited[i]:
            comp=[]
            dfs(i, comp)
            if len(comp) >= 2:
                groups.append(comp)
    if not QUIET:
        print(f"[GROUPING] Found {len(groups)} groups (size >= 2)")
    cross_standard_groups=[]
    for g in groups:
        standards={clauses[idx].standard_name for idx in g}
        if len(standards) >= 2:
            cross_standard_groups.append(g)
        else:
            if not QUIET:
                print(f"[GROUPING] Skipping same-standard group: {[clauses[idx].clause_number for idx in g]}")
    if not QUIET:
        print(f"[GROUPING] Created {len(cross_standard_groups)} cross-standard groups")
        for i,g in enumerate(cross_standard_groups):
            stds={clauses[idx].standard_name for idx in g}
            print(f"  Group {i+1}: {len(g)} clauses from {len(stds)} standards: {stds}")
    return cross_standard_groups

def group_clauses_by_category_then_similarity(clauses: List[Clause], similarity_threshold: float = 0.3, embed_fn=None) -> Tuple[List[List[int]], Dict[int,str]]:
    print(f"\n[PHASE 0+0.5+1] Grouping {len(clauses)} clauses by (category, requirement_type), then by similarity...")
    clauses = enrich_clauses_with_metadata(clauses)
    clauses_by_bucket: Dict[Tuple[str,str], List[Clause]] = defaultdict(list)
    clause_idx_by_bucket: Dict[Tuple[str,str], List[int]] = defaultdict(list)
    for idx, clause in enumerate(clauses):
        key=(clause.category or CATEGORY_MISC, clause.requirement_type or REQUIREMENT_TYPE_MISC)
        clauses_by_bucket[key].append(clause)
        clause_idx_by_bucket[key].append(idx)
    print(f"[PHASE 0+0.5] Partitioned into {len(clauses_by_bucket)} (category, requirement_type) buckets")
    all_groups: List[List[int]]=[]
    group_to_category: Dict[int,str]={}
    for (category, req_type), bucket_clauses in clauses_by_bucket.items():
        if len(bucket_clauses) < 2:
            print(f"\n[PHASE 1] Skipping bucket ({CATEGORY_TITLE_MAP.get(category,category)} / {req_type}) - only {len(bucket_clauses)} clause")
            continue
        print(f"\n[PHASE 1] Processing bucket: {CATEGORY_TITLE_MAP.get(category,category)} / {req_type} ({len(bucket_clauses)} clauses)")
        if len(bucket_clauses) == 2:
            stds={bucket_clauses[0].standard_name, bucket_clauses[1].standard_name}
            if len(stds)==2:
                print("[PHASE 1][FALLBACK] Forcing group creation for 2-clause cross-standard bucket (similarity bypass).")
                orig=clause_idx_by_bucket[(category, req_type)]
                gid=len(all_groups)
                all_groups.append(orig[:])
                group_to_category[gid]=category
                continue
        bucket_embeddings,_=compute_embeddings(bucket_clauses, embed_fn=embed_fn)
        bucket_groups=group_clauses_by_similarity(bucket_clauses, bucket_embeddings, similarity_threshold)
        orig_indices=clause_idx_by_bucket[(category, req_type)]
        for lg in bucket_groups:
            global_group=[orig_indices[i] for i in lg]
            gid=len(all_groups)
            all_groups.append(global_group)
            group_to_category[gid]=category
    print(f"\n[PHASE 1] Total groups created across all (category, requirement_type) buckets: {len(all_groups)}")
    return all_groups, group_to_category

def load_and_group_clauses(csv_paths: List[str], similarity_threshold: float = 0.3, embed_fn=None) -> Tuple[List[Clause], List[List[int]]]:
    all_clauses: List[Clause]=[]
    for p in csv_paths:
        all_clauses.extend(load_clauses_from_tagged_csv(p))
    print(f"[GROUPING] Loaded {len(all_clauses)} total clauses from {len(csv_paths)} files")
    groups,_=group_clauses_by_category_then_similarity(all_clauses, similarity_threshold=similarity_threshold, embed_fn=embed_fn)
    return all_clauses, groups
    # REAL EMBEDDINGS MODE
