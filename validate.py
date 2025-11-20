"""
Post-processing validation for GPT extracted requirements.

PHILOSOPHY: "When in doubt, include it"
Better to over-include and let manual review prune false positives than miss requirements.

MINIMAL filtering - only remove obvious non-requirements:
1. Pure definitions (Section 3) with NO requirement keywords
2. Introductory preamble ("This clause of ISO X is applicable" with no additional content)
3. Pure test methodology (no requirement language)
4. "N/A" placeholder entries
"""
import re
import csv
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# ============================================================================
# SECTION NAME MAPPING (for parent section recognition)
# ============================================================================

# Common section patterns across e-bike standards
SECTION_NAMES = {
    # Standard structural sections
    "1": "Scope",
    "2": "Normative references",
    "3": "Terms and definitions",
    "4": "General requirements",
    "5": "General requirements",
    "6": "Mechanical requirements",
    "7": "Environmental requirements",
    "8": "Safety requirements",
    "9": "Electrical requirements",
    "10": "System requirements",

    # Common annexes
    "A": "Annex A",
    "AA": "Annex AA - Rationale",
    "BB": "Annex BB - Marking and instructions",
    "CC": "Annex CC - Additional requirements",
    "DD": "Annex DD - Test procedures",
    "EE": "Annex EE - Bibliography",
    "FF": "Annex FF - Test specifications",
    "GG": "Annex GG - Test methods",
}


def parse_parent_section(clause: str) -> Optional[str]:
    """
    Parse clause number to extract parent section.

    Examples:
        "5.1.101" → "5. General requirements"
        "BB.1.1" → "Annex BB - Marking and instructions"
        "7.2.101.a" → "7. Environmental requirements"

    Returns:
        Section name with number, or None if cannot parse
    """
    clause = clause.strip()

    # Match annex patterns (AA, BB, CC, etc.)
    annex_match = re.match(r'^([A-Z]{1,2})\.', clause)
    if annex_match:
        annex_letter = annex_match.group(1)
        section_name = SECTION_NAMES.get(annex_letter, f"Annex {annex_letter}")
        return f"{annex_letter}. {section_name}" if not section_name.startswith("Annex") else section_name

    # Match numeric sections (5.1.101, 7.2, etc.)
    numeric_match = re.match(r'^(\d+)\.', clause)
    if numeric_match:
        section_num = numeric_match.group(1)
        section_name = SECTION_NAMES.get(section_num, "Unknown Section")
        return f"{section_num}. {section_name}"

    # Standalone number (1, 2, 3, etc.)
    if clause.isdigit():
        section_name = SECTION_NAMES.get(clause, "Unknown Section")
        return f"{clause}. {section_name}"

    return None


# ============================================================================
# MINIMAL FILTERING PATTERNS
# ============================================================================

# Required keywords that indicate a real requirement
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
    'provide',
    'include',
    'contain',
    'state',
    'indicate',
]


def is_pure_definition(clause: str, text: str) -> Tuple[bool, str]:
    """
    Check if this is a pure definition (Section 3) with NO requirement language.

    Returns:
        (is_pure_definition, reason)
    """
    # Only check Section 3 items
    if not re.match(r'^3(\.\d+)*$', clause.strip()):
        return False, ""

    # If it has requirement keywords, keep it (not a pure definition)
    text_lower = text.lower()
    for keyword in REQUIRED_KEYWORDS:
        if keyword in text_lower:
            return False, ""

    # Pure definition - no requirement language
    return True, "Pure definition in Section 3 (no requirement keywords)"


def is_preamble(text: str) -> Tuple[bool, str]:
    """
    Check if this is introductory preamble with no additional content.

    Examples:
        "This clause of ISO 12405-3:2014 is applicable"
        "This section applies to all batteries"

    Returns:
        (is_preamble, reason)
    """
    text = text.strip()

    # Very short preamble patterns
    preamble_patterns = [
        r'^This clause of .+ is applicable\.?$',
        r'^This section of .+ applies\.?$',
        r'^Not applicable\.?$',
        r'^See .+\.$',
    ]

    for pattern in preamble_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True, f"Introductory preamble: {pattern}"

    return False, ""


def is_na_placeholder(text: str) -> Tuple[bool, str]:
    """
    Check if this is just "N/A" or similar placeholder.

    Returns:
        (is_placeholder, reason)
    """
    text = text.strip().upper()

    if text in ['N/A', 'NA', 'NOT APPLICABLE', 'NONE', '-']:
        return True, "N/A placeholder"

    return False, ""


def is_pure_test_methodology(text: str) -> Tuple[bool, str]:
    """
    Check if this is pure test methodology with NO requirement language.

    Examples:
        "Test T.1: Connect equipment to power supply and measure voltage"
        "Number of samples: 5"

    Returns:
        (is_pure_test, reason)
    """
    text_lower = text.lower()

    # If it has requirement keywords, keep it (test requirements are valid)
    for keyword in REQUIRED_KEYWORDS:
        if keyword in text_lower:
            return False, ""

    # Pure methodology patterns (no requirement language)
    test_patterns = [
        r'Test T\.\d+:',
        r'Number of samples?:',
        r'Test equipment:',
        r'Measurement procedure:',
        r'Connect .+ and measure',
        r'Apply .+ and observe',
    ]

    for pattern in test_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, f"Pure test methodology: {pattern}"

    return False, ""


def has_minimum_substance(text: str) -> bool:
    """
    Check if text has minimum substance (not just a heading).

    Very lenient - only reject extremely short items.
    """
    text = text.strip()

    # Must have at least 10 characters
    if len(text) < 10:
        return False

    # Must have at least 2 words
    if len(text.split()) < 2:
        return False

    return True


def clean_comments_field(req: Dict) -> Dict:
    """
    Clean up the Comments field if it contains extraction metadata.

    Removes "GPT extraction" and similar metadata.
    """
    if 'Comments' in req:
        comment = req['Comments'].strip()

        # Remove extraction metadata
        if comment.lower() in ['gpt extraction', 'ai extraction', 'automated extraction']:
            req['Comments'] = ''

        # Remove "N/A" placeholders
        if comment.upper() in ['N/A', 'NA']:
            req['Comments'] = ''

    return req


def validate_requirement(req: Dict) -> Tuple[bool, str]:
    """
    Validate a single requirement using MINIMAL filtering.

    PHILOSOPHY: When in doubt, include it.

    Returns:
        (is_valid, rejection_reason)
    """
    clause = req.get('clause', '').strip()
    text = req.get('text', '').strip()

    # Clean Comments field
    req = clean_comments_field(req)

    # Check minimal exclusion filters

    # 1. N/A placeholders
    is_na, reason = is_na_placeholder(text)
    if is_na:
        return False, reason

    # 2. Preamble
    is_pream, reason = is_preamble(text)
    if is_pream:
        return False, reason

    # 3. Pure definitions (Section 3 with no requirement keywords)
    is_def, reason = is_pure_definition(clause, text)
    if is_def:
        return False, reason

    # 4. Pure test methodology (no requirement keywords)
    is_test, reason = is_pure_test_methodology(text)
    if is_test:
        return False, reason

    # 5. Minimum substance check (very lenient)
    if not has_minimum_substance(text):
        return False, f"Too short or lacks substance ({len(text)} chars)"

    # 6. Try to populate parent section if missing
    if 'Parent Section' in req:
        if req.get('Parent Section', '').strip() in ['', 'Unknown', 'N/A']:
            parent = parse_parent_section(clause)
            if parent:
                req['Parent Section'] = parent

    # Passed all filters - INCLUDE IT
    return True, ""


# ============================================================================
# MAIN VALIDATION FUNCTIONS
# ============================================================================

def validate_requirements(
    requirements: List[Dict],
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Filter GPT-extracted requirements using MINIMAL filtering.

    Args:
        requirements: List of {clause, text} dicts (may have additional fields)
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
        print("VALIDATION RESULTS (MINIMAL FILTERING)")
        print("=" * 80)
        print(f"Input:   {stats['total_input']} requirements")
        print(f"Valid:   {stats['valid_output']} requirements ({stats['valid_output']/stats['total_input']*100:.1f}%)")
        print(f"Removed: {stats['removed_total']} requirements ({stats['removed_total']/stats['total_input']*100:.1f}%)")
        print()

        if stats['removal_reasons']:
            print("Removal breakdown:")
            for reason, count in sorted(stats['removal_reasons'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count}")
        else:
            print("No items removed - all items passed minimal filtering")

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
        fieldnames = reader.fieldnames
        for row in reader:
            requirements.append(row)

    # Validate
    valid, removed, stats = validate_requirements(requirements, verbose=verbose)

    # Write valid requirements
    if valid:
        # Preserve all original fields
        output_fieldnames = fieldnames if fieldnames else ['clause', 'text']

        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(valid)

        if verbose:
            print(f"\n[OK] Valid requirements saved to: {output_csv}")

    # Write removed requirements
    if removed:
        # Add removal_reason to fieldnames
        removed_fieldnames = list(fieldnames if fieldnames else ['clause', 'text'])
        if 'removal_reason' not in removed_fieldnames:
            removed_fieldnames.append('removal_reason')

        with open(removed_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=removed_fieldnames, extrasaction='ignore')
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
        print("\nMINIMAL FILTERING - 'When in doubt, include it'")
        print("Only removes:")
        print("  - Pure definitions (Section 3 with no requirement keywords)")
        print("  - Introductory preamble")
        print("  - Pure test methodology (no requirement language)")
        print("  - N/A placeholders")
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
