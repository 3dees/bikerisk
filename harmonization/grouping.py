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
# Phase 0.7: Staged Clustering Configuration (large connected components)
# =============================================================================
# Components (within a bucket) whose size >= LARGE_COMPONENT_SIZE will be
# decomposed via staged clustering using progressively lower similarity thresholds.
# These thresholds are independent of the bucket-level dynamic threshold logic.
LARGE_COMPONENT_SIZE = 150  # minimum size to trigger staged clustering
CORE_SIM_THRESHOLD = 0.55   # core groups must meet or exceed this similarity
EXPAND_SIM_THRESHOLD = 0.50 # expansion attaches moderately similar neighbors
ATTACH_SIM_THRESHOLD = 0.45 # final attachment threshold (aligns with large-bucket threshold)


# =============================================================================
# PHASE 0.5: REQUIREMENT TYPE CONSTANTS
# =============================================================================

REQUIREMENT_TYPE_DEFINITIONS = "definitions"
REQUIREMENT_TYPE_SCOPE = "scope"
REQUIREMENT_TYPE_NORMATIVE_REFERENCES = "normative_references"

REQUIREMENT_TYPE_TEST_METHOD = "test_method"
REQUIREMENT_TYPE_TEST_SETUP = "test_setup"
REQUIREMENT_TYPE_SAMPLE_SIZE = "sample_size"
REQUIREMENT_TYPE_TEST_ACCEPTANCE = "test_acceptance"

REQUIREMENT_TYPE_PROTECTIVE_DEVICES = "protective_devices"
REQUIREMENT_TYPE_BMS_BEHAVIOR = "bms_behavior"

REQUIREMENT_TYPE_CONSTRUCTION_REQUIREMENTS = "construction_requirements"
REQUIREMENT_TYPE_MECHANICAL_STRESS = "mechanical_stress"
REQUIREMENT_TYPE_THERMAL_BEHAVIOR = "thermal_behavior"
REQUIREMENT_TYPE_ELECTRICAL_REQUIREMENTS = "electrical_requirements"
REQUIREMENT_TYPE_PERFORMANCE_LIMITS = "performance_limits"

REQUIREMENT_TYPE_LABELING = "labeling"
REQUIREMENT_TYPE_WARNINGS = "warnings"
REQUIREMENT_TYPE_USER_INSTRUCTIONS = "user_instructions"
REQUIREMENT_TYPE_CHARGING_STORAGE = "charging_storage"
REQUIREMENT_TYPE_PACKAGING_TRANSPORT = "packaging_transport"

REQUIREMENT_TYPE_ENVIRONMENTAL = "environmental"
REQUIREMENT_TYPE_DISPOSAL_RECYCLING = "disposal_recycling"

REQUIREMENT_TYPE_GENERAL_SAFETY = "general_safety_requirements"
REQUIREMENT_TYPE_PROCEDURAL_GENERAL = "procedural_general"
REQUIREMENT_TYPE_ADMINISTRATIVE = "administrative_requirements"

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
import json
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
    """
    Phase 0.5 + 0.6: expanded, deterministic requirement-type classifier.

    Uses:
      - clause.text (lowercased)
      - clause.clause_number (if present)

    Ordering matters:
      - More specific patterns appear before more generic ones.
      - REQUIREMENT_TYPE_MISC is a true fallback.
    """
    text = (clause.text or "").lower()
    cid = (getattr(clause, "clause_number", "") or "").lower()

    # ------------------------------------------------------------------
    # 1. DEFINITIONS / SCOPE / NORMATIVE REFERENCES
    # ------------------------------------------------------------------

    # Normative references (often clause 2.x in IEC/EN)
    if "normative reference" in text or "normative references" in text or cid.startswith("2."):
        return REQUIREMENT_TYPE_NORMATIVE_REFERENCES

    # Definitions (often clause 3.x)
    if "terms and definitions" in text or "definition" in text or "definitions" in text or cid.startswith("3."):
        return REQUIREMENT_TYPE_DEFINITIONS

    # Scope (often clause 1.x)
    if "scope" in text and ("this standard" in text or "applies to" in text or cid.startswith("1.")):
        return REQUIREMENT_TYPE_SCOPE

    # ------------------------------------------------------------------
    # 2. DISPOSAL / RECYCLING
    # ------------------------------------------------------------------
    if any(kw in text for kw in [
        "disposal",
        "dispose of",
        "disposed of",
        "recycle",
        "recycling",
        "end-of-life",
        "end of life",
        "waste electrical and electronic equipment",
        "weee",
    ]):
        return REQUIREMENT_TYPE_DISPOSAL_RECYCLING

    # ------------------------------------------------------------------
    # 3. PACKAGING / TRANSPORT vs CHARGING / STORAGE
    # ------------------------------------------------------------------

    # Packaging / transport
    if any(kw in text for kw in [
        "packaging",
        "package shall",
        "packaged batteries",
        "transport",
        "transportation",
        "shipping",
        "shipment",
        "carriage",
        "packing group",
    ]):
        return REQUIREMENT_TYPE_PACKAGING_TRANSPORT

    # Charging / storage
    if any(kw in text for kw in [
        "charging",
        "charge the battery",
        "recharge",
        "charger",
        "charging temperature",
        "storage temperature",
        "store the battery",
        "storage conditions",
        "state of charge",
    ]):
        return REQUIREMENT_TYPE_CHARGING_STORAGE

    # ------------------------------------------------------------------
    # 4. TEST-RELATED: SAMPLE SIZE, SETUP, METHOD, ACCEPTANCE
    # ------------------------------------------------------------------

    # Sample size
    if any(kw in text for kw in [
        "sample size",
        "samples shall be",
        "number of samples",
        "at least three samples",
        "at least 3 samples",
        "shall be tested in groups of",
    ]):
        return REQUIREMENT_TYPE_SAMPLE_SIZE

    # Test setup (fixtures, orientation, conditioning)
    if any(kw in text for kw in [
        "test setup",
        "test set-up",
        "mounted on",
        "mounted in",
        "fixed to",
        "arranged as in normal use",
        "arranged in normal use",
        "preconditioning",
        "pre-conditioning",
        "conditioned at",
        "placed in a chamber",
    ]):
        return REQUIREMENT_TYPE_TEST_SETUP

    # Test method / procedure
    if any(kw in text for kw in [
        "test method",
        "test procedure",
        "the following test shall be carried out",
        "the following test shall be conducted",
        "the test is carried out",
        "the test is performed",
        "is subjected to the following",
        "is tested as follows",
    ]):
        return REQUIREMENT_TYPE_TEST_METHOD

    # Test acceptance criteria (explicit test outcomes)
    if any(kw in text for kw in [
        "no rupture shall occur",
        "no fire shall occur",
        "no explosion shall occur",
        "no leakage shall occur",
        "shall not rupture",
        "shall not explode",
        "shall not catch fire",
        "shall remain within the specified limits",
        "criteria are fulfilled",
        "compliance is checked by",
        "the test is considered satisfactory if",
    ]):
        return REQUIREMENT_TYPE_TEST_ACCEPTANCE

    # ------------------------------------------------------------------
    # 5. PROTECTIVE DEVICES / BMS / PROTECTION LOGIC
    # ------------------------------------------------------------------

    if any(kw in text for kw in [
        "protection device",
        "protective device",
        "protective devices",
        "fuse",
        "fuses",
        "thermal cut-out",
        "thermal cutoff",
        "resettable device",
        "circuit breaker",
        "protective circuit",
    ]):
        return REQUIREMENT_TYPE_PROTECTIVE_DEVICES

    if any(kw in text for kw in [
        "battery management system",
        "bms",
        "protection circuit",
        "protection circuitry",
        "overcharge protection",
        "over-discharge protection",
        "overdischarge protection",
        "overcurrent protection",
        "short-circuit protection",
        "cell balancing",
        "cut-off device",
        "cutoff device",
        "monitoring of cell voltage",
        "monitoring of temperature",
    ]):
        return REQUIREMENT_TYPE_BMS_BEHAVIOR

    # ------------------------------------------------------------------
    # 6. CONSTRUCTION / MECHANICAL / THERMAL / ELECTRICAL
    # ------------------------------------------------------------------

    # Thermal behavior
    if any(kw in text for kw in [
        "temperature rise",
        "surface temperature",
        "max temperature",
        "maximum temperature",
        "overheating",
        "thermal runaway",
        "flammable gas",
        "venting",
        "venting mechanism",
    ]):
        return REQUIREMENT_TYPE_THERMAL_BEHAVIOR

    # Mechanical stress (drop, crush, vibration, impact)
    if any(kw in text for kw in [
        "drop test",
        "impact test",
        "shock test",
        "vibration test",
        "crush test",
        "mechanical abuse",
        "mechanical shock",
        "falling mass",
        "free fall",
        "torsion",
        "bending",
        "compression force",
    ]):
        return REQUIREMENT_TYPE_MECHANICAL_STRESS

    # Electrical requirements
    if any(kw in text for kw in [
        "short circuit",
        "short-circuit",
        "internal resistance",
        "insulation resistance",
        "dielectric strength",
        "clearance",
        "creepage distance",
        "protective earth",
        "earthing",
        "insulation",
    ]):
        return REQUIREMENT_TYPE_ELECTRICAL_REQUIREMENTS

    # Construction-oriented requirements
    if any(kw in text for kw in [
        "shall be constructed",
        "shall be so constructed",
        "mechanical construction",
        "enclosure shall",
        "enclosures shall",
        "enclosure material",
        "fixing means",
        "mechanical strength",
        "degree of protection",
        "ingress protection",
        "ip code",
    ]):
        return REQUIREMENT_TYPE_CONSTRUCTION_REQUIREMENTS

    # ------------------------------------------------------------------
    # 7. PERFORMANCE LIMITS / RATINGS
    # ------------------------------------------------------------------

    if any(kw in text for kw in [
        "rated voltage",
        "rated capacity",
        "rated current",
        "current rating",
        "maximum charge current",
        "maximum discharge current",
        "maximum continuous current",
        "limits shall not be exceeded",
        "operating range",
        "specified range",
    ]):
        return REQUIREMENT_TYPE_PERFORMANCE_LIMITS

    # ------------------------------------------------------------------
    # 8. DOCUMENTATION: LABELING / WARNINGS / INSTRUCTIONS
    # ------------------------------------------------------------------

    # Labeling / markings
    if any(kw in text for kw in [
        "shall be marked",
        "marking",
        "markings",
        "label",
        "labels",
        "rating plate",
        "symbol",
        "symbols",
        "pictogram",
    ]):
        return REQUIREMENT_TYPE_LABELING

    # Warnings
    if any(kw in text for kw in [
        "warning:",
        "warnings shall",
        "caution:",
        "risk of",
        "hazard",
        "danger of",
        "to reduce the risk",
    ]):
        return REQUIREMENT_TYPE_WARNINGS

    # User instructions (non-charging/storage)
    if any(kw in text for kw in [
        "instructions for use",
        "instructions shall",
        "user instructions",
        "the instructions shall",
        "instruction manual",
        "accompanying documents",
        "information for the user",
    ]):
        if any(kw in text for kw in [
            "charging",
            "charge the battery",
            "recharge",
            "storage temperature",
            "store the battery",
        ]):
            return REQUIREMENT_TYPE_CHARGING_STORAGE
        return REQUIREMENT_TYPE_USER_INSTRUCTIONS

    # ------------------------------------------------------------------
    # 9. ENVIRONMENTAL CONDITIONS (non-test-specific)
    # ------------------------------------------------------------------

    if any(kw in text for kw in [
        "ambient temperature",
        "relative humidity",
        "humidity",
        "altitude",
        "environmental conditions",
        "pollution degree",
    ]):
        return REQUIREMENT_TYPE_ENVIRONMENTAL

    # ------------------------------------------------------------------
    # 10. GENERIC SAFETY / PROCEDURAL / ADMINISTRATIVE (Phase 0.6)
    # ------------------------------------------------------------------

    # Generic safety requirements (broad safety / risk language)
    if any(kw in text for kw in [
        "shall not present a risk",
        "shall not present any risk",
        "shall not cause injury",
        "shall not cause fire",
        "shall not constitute a hazard",
        "shall not be hazardous",
        "safe use",
        "safe operation",
        "safety of the user",
        "protection against",
        "protection of persons",
        "reduce the risk of injury",
        "reduce the risk of fire",
    ]):
        return REQUIREMENT_TYPE_GENERAL_SAFETY

    # General procedural text (applies to, in accordance with, compliance logic)
    if any(kw in text for kw in [
        "applies to all",
        "applies to the following",
        "unless otherwise specified",
        "except as specified",
        "as specified in",
        "as described in",
        "in accordance with",
        "according to",
        "shall comply with",
        "compliance shall be checked",
        "compliance is determined by",
        "compliance is verified by",
        "compliance is achieved when",
    ]):
        return REQUIREMENT_TYPE_PROCEDURAL_GENERAL

    # Administrative / documentation (non-user-facing, technical/traceability)
    if any(kw in text for kw in [
        "technical documentation",
        "documentation shall",
        "shall be documented",
        "test report",
        "test reports",
        "type test report",
        "records shall be kept",
        "record shall be kept",
        "record of",
        "manufacturer shall provide",
        "information to be provided",
        "data sheet",
        "datasheet",
    ]):
        return REQUIREMENT_TYPE_ADMINISTRATIVE

    # ------------------------------------------------------------------
    # 11. FINAL FALLBACK
    # ------------------------------------------------------------------

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

def _print_similarity_stats(similarity_matrix: np.ndarray, bucket_name: str):
    """Print aggregated similarity statistics (no full matrix). Always includes requested percentiles and edge counts."""
    n = similarity_matrix.shape[0]
    if n < 2:
        return
    tri_indices = np.triu_indices(n, k=1)
    sims = similarity_matrix[tri_indices]
    if sims.size == 0:
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
        '>=0.40': int(np.sum(sims >= 0.40)),
        '>=0.45': int(np.sum(sims >= 0.45)),
    }
    bins = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(sims, bins=bins)
    bucket_ranges = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
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


def staged_cluster_large_component(component_indices, sim_matrix: np.ndarray):
    """Phase 0.7 staged clustering for a large connected component.

    Args:
        component_indices (List[int]): indices (local to bucket) of the large component.
        sim_matrix (np.ndarray): full similarity matrix for the bucket (square n x n).

    Returns:
        List[List[int]]: list of subclusters (indices local to bucket).
    """
    size = len(component_indices)
    if size < LARGE_COMPONENT_SIZE:
        return [list(component_indices)]  # no staging needed

    # Build local sub-matrix view for convenience
    # Map local position -> original bucket index
    local_index_map = {orig_idx: i for i, orig_idx in enumerate(component_indices)}
    sub_sim = sim_matrix[np.ix_(component_indices, component_indices)]
    n = sub_sim.shape[0]

    # Step 1: Identify core groups using CORE_SIM_THRESHOLD
    core_adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if sub_sim[i, j] >= CORE_SIM_THRESHOLD:
                core_adj[i].append(j)
                core_adj[j].append(i)

    visited = [False] * n
    cores = []
    def dfs_core(node, comp):
        visited[node] = True
        comp.append(node)
        for nb in core_adj[node]:
            if not visited[nb]:
                dfs_core(nb, comp)
    for i in range(n):
        if not visited[i]:
            comp = []
            dfs_core(i, comp)
            if len(comp) >= 2:
                cores.append(comp)

    if not cores:
        # Fallback: treat entire component as one cluster if no dense cores found
        return [list(component_indices)]

    # Track assignment of nodes to core clusters
    node_to_core = {}
    for core_id, core_nodes in enumerate(cores):
        for ln in core_nodes:
            node_to_core[ln] = core_id

    # Step 2: Expansion at EXPAND_SIM_THRESHOLD
    for ln in range(n):
        if ln in node_to_core:
            continue
        best_core = None
        best_sim = -1.0
        for core_id, core_nodes in enumerate(cores):
            # Compute max similarity to any member of core
            max_sim = np.max(sub_sim[ln, core_nodes])
            if max_sim >= EXPAND_SIM_THRESHOLD and max_sim > best_sim:
                best_sim = max_sim
                best_core = core_id
        if best_core is not None:
            cores[best_core].append(ln)
            node_to_core[ln] = best_core

    # Step 3: Attachment at ATTACH_SIM_THRESHOLD
    for ln in range(n):
        if ln in node_to_core:
            continue
        best_core = None
        best_sim = -1.0
        for core_id, core_nodes in enumerate(cores):
            max_sim = np.max(sub_sim[ln, core_nodes])
            if max_sim >= ATTACH_SIM_THRESHOLD and max_sim > best_sim:
                best_sim = max_sim
                best_core = core_id
        if best_core is not None:
            cores[best_core].append(ln)
            node_to_core[ln] = best_core

    # Any still-unassigned nodes become singleton micro-clusters
    clusters = []
    for core_nodes in cores:
        clusters.append([component_indices[ln] for ln in core_nodes])
    for ln in range(n):
        if ln not in node_to_core:
            clusters.append([component_indices[ln]])
    return clusters

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
    components: List[List[int]] = []
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
                components.append(comp)
    if not QUIET:
        print(f"[GROUPING] Found {len(components)} connected components (size >= 2)")

    # Phase 0.7: apply staged clustering to very large components
    staged_groups: List[List[int]] = []
    for comp in components:
        if len(comp) >= LARGE_COMPONENT_SIZE:
            if not QUIET:
                print(f"[GROUPING][STAGE] Applying staged clustering to large component size={len(comp)}")
            subclusters = staged_cluster_large_component(comp, similarity_matrix)
            for sc in subclusters:
                if len(sc) >= 2:
                    staged_groups.append(sc)
        else:
            staged_groups.append(comp)

    cross_standard_groups=[]
    for g in staged_groups:
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


def summarize_cluster_groups(groups, max_examples_per_group: int = 5) -> List[dict]:
    """Return a lightweight summary of each group for debugging.

    Accepts either:
    - Iterable of objects with attribute 'clauses' (each item Clause), or
    - Iterable of iterables of Clause instances.

    Returns list of dicts with fields:
    - group_id
    - category
    - requirement_type
    - size
    - standards_present
    - example_clause_ids
    - example_snippets
    """
    summaries: List[dict] = []
    for gid, g in enumerate(groups, start=1):
        # Resolve clauses sequence
        clauses_seq = None
        if hasattr(g, 'clauses'):
            clauses_seq = getattr(g, 'clauses')
        elif isinstance(g, (list, tuple)) and g and hasattr(g[0], 'text'):
            clauses_seq = g
        else:
            # Unsupported group type; skip gracefully
            continue

        size = len(clauses_seq)
        if size == 0:
            continue

        # Derive category / requirement_type from majority vote if present
        categories = [getattr(c, 'category', None) for c in clauses_seq]
        req_types = [getattr(c, 'requirement_type', None) for c in clauses_seq]
        cat = None
        if any(categories):
            counts = {}
            for c in categories:
                if c:
                    counts[c] = counts.get(c, 0) + 1
            if counts:
                cat = max(counts.items(), key=lambda kv: kv[1])[0]
        req_t = None
        if any(req_types):
            counts = {}
            for r in req_types:
                if r:
                    counts[r] = counts.get(r, 0) + 1
            if counts:
                req_t = max(counts.items(), key=lambda kv: kv[1])[0]

        # Standards present
        standards = sorted({getattr(c, 'standard_name', None) for c in clauses_seq if getattr(c, 'standard_name', None)})

        # Examples
        examples = clauses_seq[:max_examples_per_group]
        example_ids = [getattr(c, 'clause_number', None) or getattr(c, 'id', None) for c in examples]
        example_snippets = []
        for c in examples:
            text = getattr(c, 'text', '') or ''
            snippet = text.replace('\n', ' ')[:120]
            example_snippets.append(snippet)

        summaries.append({
            'group_id': gid,
            'category': cat,
            'requirement_type': req_t,
            'size': size,
            'standards_present': standards,
            'example_clause_ids': example_ids,
            'example_snippets': example_snippets,
        })
    return summaries

def group_clauses_by_category_then_similarity(clauses: List[Clause], similarity_threshold: float = 0.3, embed_fn=None) -> Tuple[List[List[int]], Dict[int,str]]:
    """
    Group clauses by (category, requirement_type) then perform graph-based clustering
    with dynamic similarity thresholds:

    - Buckets with size <= 100 use base threshold (similarity_threshold, default 0.30)
    - Buckets with size > 100 use elevated threshold (0.40) to prevent mega-cluster collapse

    All other behavior (fallback for 2-clause cross-standard buckets, multi-standard filter) is unchanged.
    """
    print(f"\n[PHASE 0+0.5+1] Grouping {len(clauses)} clauses by (category, requirement_type), then by similarity (dynamic thresholds)...")
    clauses = enrich_clauses_with_metadata(clauses)
    clauses_by_bucket: Dict[Tuple[str,str], List[Clause]] = defaultdict(list)
    clause_idx_by_bucket: Dict[Tuple[str,str], List[int]] = defaultdict(list)
    for idx, clause in enumerate(clauses):
        key = (clause.category or CATEGORY_MISC, clause.requirement_type or REQUIREMENT_TYPE_MISC)
        clauses_by_bucket[key].append(clause)
        clause_idx_by_bucket[key].append(idx)
    print(f"[PHASE 0+0.5] Partitioned into {len(clauses_by_bucket)} (category, requirement_type) buckets")

    LARGE_BUCKET_THRESHOLD = 100
    HIGH_THRESHOLD = 0.45  # elevated threshold for large buckets (>100 clauses) to reduce mega-components

    all_groups: List[List[int]] = []
    group_to_category: Dict[int,str] = {}

    # Diagnostics toggle: enable JSON summaries for specific buckets via env var
    DEBUG_CLUSTER_SUMMARY = os.environ.get("CLUSTER_DEBUG_SUMMARY", "0") == "1"

    for (category, req_type), bucket_clauses in clauses_by_bucket.items():
        bucket_size = len(bucket_clauses)
        if bucket_size < 2:
            print(f"\n[PHASE 1] Skipping bucket ({CATEGORY_TITLE_MAP.get(category,category)} / {req_type}) - only {bucket_size} clause")
            continue
        # Dynamic threshold selection
        bucket_threshold = similarity_threshold if bucket_size <= LARGE_BUCKET_THRESHOLD else HIGH_THRESHOLD
        print(f"\n[PHASE 1] Processing bucket: {CATEGORY_TITLE_MAP.get(category,category)} / {req_type} ({bucket_size} clauses) | threshold={bucket_threshold:.2f}")

        # Fallback for 2-clause cross-standard bucket remains unchanged
        if bucket_size == 2:
            stds = {bucket_clauses[0].standard_name, bucket_clauses[1].standard_name}
            if len(stds) == 2:
                print("[PHASE 1][FALLBACK] Forcing group creation for 2-clause cross-standard bucket (similarity bypass).")
                orig = clause_idx_by_bucket[(category, req_type)]
                gid = len(all_groups)
                all_groups.append(orig[:])
                group_to_category[gid] = category
                continue

        bucket_embeddings, _ = compute_embeddings(bucket_clauses, embed_fn=embed_fn)
        # For large buckets, compute and print aggregated similarity stats before clustering.
        # Always print similarity stats for very large buckets (>=300 clauses) regardless of QUIET flag
        if bucket_size >= 300:
            sim_matrix = cosine_similarity(bucket_embeddings)
            _print_similarity_stats(sim_matrix, f"{CATEGORY_TITLE_MAP.get(category, category)} / {req_type}")
        bucket_groups = group_clauses_by_similarity(bucket_clauses, bucket_embeddings, bucket_threshold)

        # Optional diagnostics: summarize the groups for the (misc, misc) bucket
        if DEBUG_CLUSTER_SUMMARY and (category == CATEGORY_MISC) and (req_type == REQUIREMENT_TYPE_MISC):
            # Convert index groups into sequences of Clause for summarization
            clause_groups = [[bucket_clauses[i] for i in idxs] for idxs in bucket_groups]
            summaries = summarize_cluster_groups(clause_groups, max_examples_per_group=5)
            os.makedirs("debug", exist_ok=True)
            out_path = os.path.join("debug", "misc_misc_cluster_summary.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summaries, f, indent=2, ensure_ascii=False)
                print(f"[DEBUG] Wrote cluster summary: {out_path} ({len(summaries)} groups)")
            except Exception as e:
                print(f"[DEBUG] Failed to write cluster summary: {e}")

        orig_indices = clause_idx_by_bucket[(category, req_type)]
        for lg in bucket_groups:
            global_group = [orig_indices[i] for i in lg]
            gid = len(all_groups)
            all_groups.append(global_group)
            group_to_category[gid] = category

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
