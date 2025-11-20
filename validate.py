"""
Post-processing validation for GPT extracted requirements.

Filters out garbage like definitions, test procedures, bibliography, and informative annexes.
"""
import re
import csv
from typing import List, Dict, Tuple
from pathlib import Path


# ============================================================================
# EXCLUSION PATTERNS
# ============================================================================

# Clause patterns to exclude (section numbers)
EXCLUDE_CLAUSE_PATTERNS = [
    r'^1$',  # Scope (exactly "1")
    r'^1\.\d+$',  # Scope subsections (1.1, 1.2, etc.)
    r'^2$',  # Normative references
    r'^2\.\d+$',  # Normative reference subsections
    r'^3$',  # Terms and definitions
    r'^3\.\d+(\.\d+)?$',  # Definition subsections (3.1, 3.1.1, etc.)
    r'^Bibliography$',  # Bibliography section
    r'^Annex\s+[A-Z]+',  # Annexes (will filter informative separately)
    r'^Table\s+',  # Table captions
    r'^Figure\s+',  # Figure captions
]

# Test procedure patterns
TEST_PROCEDURE_PATTERNS = [
    r'Test T\.\d+',  # UN test procedures (Test T.1, T.2, etc.)
    r'^\d+\.\d+\s+Test',  # Test sections (6.101 Test, etc.)
    r'Test\s+(item|category|sequence)',
    r'Number of samples',
    r'Purpose:\s+',  # Test purpose descriptions
    r'Requirement:\s+Cells and batteries meet',  # Test pass criteria
]

# Content patterns that indicate garbage
GARBAGE_CONTENT_PATTERNS = [
    # Definitions
    r'^[A-Za-z\s\-]+\n',  # Term followed by newline (definition format)
    r'^\w+(\s+\w+){0,5}$',  # Very short (1-6 words, likely a heading)

    # Bibliography/references
    r'^\[\d+\]',  # Bibliography entry ([1], [2], etc.)
    r'^ISO\s+\d+',  # ISO standard reference
    r'^IEC\s+\d+',  # IEC standard reference
    r'^EN\s+\d+',  # EN standard reference
    r'^UL\s+\d+',  # UL standard reference
    r'^UN\s+(ECE\s+)?Regulation',  # UN regulations

    # Table of contents
    r'\.{3,}',  # Dots pattern (TOC entries)
    r'\s+\d+$',  # Ends with page number
]

# Keywords that must appear for valid requirements
REQUIRED_KEYWORDS = [
    'shall',
    'must',
    'required',
    'instructions',
    'instruction',
    'marking',
    'label',
    'warning',
    'caution',
    'danger',
    'manual',
    'document',
    'information',
]

MIN_TEXT_LENGTH = 20  # Minimum characters for valid requirement


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def is_excluded_clause(clause: str) -> Tuple[bool, str]:
    """
    Check if clause number should be excluded.

    Returns:
        (should_exclude, reason)
    """
    clause = clause.strip()

    for pattern in EXCLUDE_CLAUSE_PATTERNS:
        if re.match(pattern, clause, re.IGNORECASE):
            return True, f"Excluded clause pattern: {pattern}"

    return False, ""


def is_test_procedure(clause: str, text: str) -> Tuple[bool, str]:
    """
    Check if this is a test procedure description.

    Returns:
        (is_test, reason)
    """
    combined = f"{clause} {text}"

    for pattern in TEST_PROCEDURE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return True, f"Test procedure pattern: {pattern}"

    return False, ""


def is_informative_annex(clause: str, text: str) -> Tuple[bool, str]:
    """
    Check if this is from an informative annex.

    Returns:
        (is_informative, reason)
    """
    combined = f"{clause} {text}"

    # Check for "(informative)" marker
    if re.search(r'\(informative\)', combined, re.IGNORECASE):
        return True, "Informative annex"

    # Specific informative annexes by letter (common: FF, GG for test specs)
    if re.match(r'^FF\.|^GG\.', clause):
        return True, "Informative test annex (FF/GG)"

    return False, ""


def has_garbage_content(text: str) -> Tuple[bool, str]:
    """
    Check if text contains garbage patterns.

    Returns:
        (is_garbage, reason)
    """
    for pattern in GARBAGE_CONTENT_PATTERNS:
        if re.search(pattern, text):
            return True, f"Garbage content pattern: {pattern}"

    return False, ""


def has_required_keywords(text: str) -> bool:
    """
    Check if text contains at least one required keyword.
    """
    text_lower = text.lower()

    for keyword in REQUIRED_KEYWORDS:
        if keyword in text_lower:
            return True

    return False


def is_too_short(text: str) -> bool:
    """
    Check if text is too short to be a real requirement.
    """
    return len(text.strip()) < MIN_TEXT_LENGTH


def validate_requirement(req: Dict) -> Tuple[bool, str]:
    """
    Validate a single requirement.

    Returns:
        (is_valid, rejection_reason)
    """
    clause = req.get('clause', '').strip()
    text = req.get('text', '').strip()

    # Check exclusion filters (order matters - most specific first)

    # 1. Informative annexes
    is_info, reason = is_informative_annex(clause, text)
    if is_info:
        return False, reason

    # 2. Test procedures
    is_test, reason = is_test_procedure(clause, text)
    if is_test:
        return False, reason

    # 3. Excluded clauses (scope, definitions, etc.)
    excluded, reason = is_excluded_clause(clause)
    if excluded:
        return False, reason

    # 4. Garbage content patterns
    is_garbage, reason = has_garbage_content(text)
    if is_garbage:
        return False, reason

    # 5. Too short
    if is_too_short(text):
        return False, f"Too short ({len(text)} chars < {MIN_TEXT_LENGTH})"

    # 6. Missing required keywords
    if not has_required_keywords(text):
        return False, "Missing required keywords (shall/must/required/etc.)"

    # Passed all filters
    return True, ""


# ============================================================================
# MAIN VALIDATION FUNCTIONS
# ============================================================================

def validate_requirements(
    requirements: List[Dict],
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Filter GPT-extracted requirements to remove garbage.

    Args:
        requirements: List of {clause, text} dicts
        verbose: Print detailed filtering stats

    Returns:
        (valid_requirements, removed_requirements, stats_dict)
    """
    valid = []
    removed = []

    # Stats tracking
    stats = {
        'total_input': len(requirements),
        'valid_output': 0,
        'removed_total': 0,
        'removal_reasons': {},
    }

    for req in requirements:
        is_valid, reason = validate_requirement(req)

        if is_valid:
            valid.append(req)
        else:
            # Track removal reason
            removed.append({
                **req,
                'removal_reason': reason
            })

            # Update stats
            if reason not in stats['removal_reasons']:
                stats['removal_reasons'][reason] = 0
            stats['removal_reasons'][reason] += 1

    stats['valid_output'] = len(valid)
    stats['removed_total'] = len(removed)

    if verbose:
        print("=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"Input:   {stats['total_input']} requirements")
        print(f"Valid:   {stats['valid_output']} requirements ({stats['valid_output']/stats['total_input']*100:.1f}%)")
        print(f"Removed: {stats['removed_total']} requirements ({stats['removed_total']/stats['total_input']*100:.1f}%)")
        print()
        print("Removal breakdown:")
        for reason, count in sorted(stats['removal_reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {reason}: {count}")
        print("=" * 80)

    return valid, removed, stats


def validate_csv_file(
    input_csv: str,
    output_csv: str = None,
    removed_csv: str = None,
    verbose: bool = True
) -> Dict:
    """
    Validate requirements from CSV file.

    Args:
        input_csv: Path to input CSV (with 'clause' and 'text' columns)
        output_csv: Path to output filtered CSV (default: input_filtered.csv)
        removed_csv: Path to removed items CSV (default: input_removed.csv)
        verbose: Print detailed stats

    Returns:
        Validation stats dict
    """
    input_path = Path(input_csv)

    if output_csv is None:
        output_csv = input_path.parent / f"{input_path.stem}_filtered.csv"
    if removed_csv is None:
        removed_csv = input_path.parent / f"{input_path.stem}_removed.csv"

    # Read input CSV
    requirements = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            requirements.append(row)

    # Validate
    valid, removed, stats = validate_requirements(requirements, verbose=verbose)

    # Write valid requirements
    if valid:
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['clause', 'text'])
            writer.writeheader()
            writer.writerows(valid)

        if verbose:
            print(f"\n[OK] Valid requirements saved to: {output_csv}")

    # Write removed requirements
    if removed:
        with open(removed_csv, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['clause', 'text', 'removal_reason']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(removed)

        if verbose:
            print(f"[REMOVED] Removed items saved to: {removed_csv}")

    return stats


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate.py <input_csv> [output_csv] [removed_csv]")
        print("\nExample:")
        print("  python validate.py test_ss_en_50604_results.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    removed_csv = sys.argv[3] if len(sys.argv) > 3 else None

    stats = validate_csv_file(input_csv, output_csv, removed_csv, verbose=True)

    # Exit code based on results
    if stats['valid_output'] > 0:
        sys.exit(0)
    else:
        print("\n[ERROR] No valid requirements found!")
        sys.exit(1)
