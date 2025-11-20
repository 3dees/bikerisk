"""
Post-processing validation for GPT extracted requirements.

PHILOSOPHY: "When in doubt, include it"
Better to over-include and let manual review prune false positives than miss requirements.

MINIMAL filtering - only remove obvious non-requirements:
1. Pure definitions (Section 3) with NO requirement keywords
2. Introductory preamble ("This clause of ISO X is applicable" with no additional content)
3. Pure test methodology (no requirement language)
4. "N/A" placeholder entries

SUCCESS CRITERIA (based on manual ground truth):
- SS_EN_50604: 142 clauses (manual count) → expect 130-150 after minimal filtering
- Key: BB.2 has 32 sub-bullets (a-ff) for instruction requirements
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
    Parse clause number to extract parent section using structure-based logic.

    PHILOSOPHY: Extract parent from clause STRUCTURE, not hardcoded mappings.
    Works across all standards (ISO, IEC, UL, EN, ANSI) automatically.

    Rules:
    - 1 segment ("5", "BB") → IS a top-level parent → return as-is
    - 2 segments ("5.101", "BB.1", "46.8") → IS a mid-level parent → return as-is
    - 3+ segments ("5.101.1", "BB.1.2") → Parent = first 2 segments

    Examples:
        "5.101.1" → "5.101"
        "BB.1.1" → "BB.1"
        "7.2.3.4" → "7.2"
        "46.8" → "46.8" (already parent level)
        "A.2.1" → "A.2"
        "5" → "5" (top-level)

    Returns:
        Parent section number, or None if cannot parse
    """
    clause = clause.strip()

    # Handle special cases
    if clause.lower().startswith('annex'):
        # "Annex GG" → "Annex GG"
        annex_match = re.match(r'annex\s+([A-Z]{1,2})', clause, re.IGNORECASE)
        if annex_match:
            return f"Annex {annex_match.group(1).upper()}"
        return "Annex"

    if clause.lower().startswith('table'):
        # "Table GG.1" → "Table GG"
        table_match = re.match(r'table\s+([A-Z]{1,2})', clause, re.IGNORECASE)
        if table_match:
            return f"Table {table_match.group(1).upper()}"
        return "Table"

    if clause.lower() == 'bibliography':
        return "Bibliography"

    # Parse clause structure: "5.101.1.a" → ["5", "101", "1", "a"]
    # Treat annex letters as single segment: "BB.1.1" → ["BB", "1", "1"]

    # First, handle letter prefixes (annexes)
    letter_prefix_match = re.match(r'^([A-Z]{1,2})\.(.+)$', clause)
    if letter_prefix_match:
        prefix = letter_prefix_match.group(1)
        rest = letter_prefix_match.group(2)
        segments = [prefix] + rest.split('.')
    else:
        # Pure numeric or mixed
        segments = clause.split('.')

    # Remove letter suffixes from last segment if present
    # "5.101.1.a" → segments = ["5", "101", "1", "a"]
    # We want numeric segments only for depth calculation
    numeric_segments = []
    for seg in segments:
        # Remove trailing letters: "1a" → "1", "101a" → "101"
        numeric_part = re.match(r'^(\d+|[A-Z]{1,2})', seg)
        if numeric_part:
            numeric_segments.append(numeric_part.group(1))

    if not numeric_segments:
        return None

    # Apply depth rules:
    # - 1 segment → return as-is (top-level parent)
    # - 2 segments → return as-is (mid-level parent, use this for grouping)
    # - 3+ segments → return first 2 segments

    if len(numeric_segments) <= 2:
        return '.'.join(numeric_segments)
    else:
        return '.'.join(numeric_segments[:2])


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


# ============================================================================
# CLASSIFICATION TAGS (Mandate Level, Safety, Manual)
# ============================================================================

def classify_mandate_level(text: str) -> str:
    """
    Classify requirement by mandate level based on normative keywords.

    Returns:
        "High" - SHALL/MUST (mandatory compliance)
        "Medium" - SHOULD/RECOMMENDED (recommended but not required)
        "Informative" - MAY/CAN or no normative keywords
    """
    text_lower = text.lower()

    # High priority - mandatory requirements
    if 'shall' in text_lower or 'must' in text_lower or 'required' in text_lower:
        return 'High'

    # Medium priority - recommendations
    if 'should' in text_lower or 'recommended' in text_lower:
        return 'Medium'

    # Informative - optional or no normative language
    return 'Informative'


def classify_safety_flag(text: str) -> str:
    """
    Flag requirements related to safety hazards.

    Returns:
        "y" if contains safety-related keywords
        "n" otherwise
    """
    text_lower = text.lower()

    safety_keywords = [
        'injury', 'injuries', 'harm', 'hazard', 'hazards', 'danger', 'dangerous',
        'fire', 'explosion', 'burn', 'electric shock', 'electrocution',
        'death', 'fatal', 'fatality',
        'risk', 'unsafe', 'safety',
        'entrapment', 'crush', 'cut', 'laceration',
        'toxic', 'poisoning', 'asphyxiation',
        'radiation', 'electromagnetic',
    ]

    for keyword in safety_keywords:
        if keyword in text_lower:
            return 'y'

    return 'n'


def classify_manual_flag(text: str) -> str:
    """
    Flag requirements related to user manuals and documentation.

    Returns:
        "y" if contains manual/documentation keywords
        "n" otherwise
    """
    text_lower = text.lower()

    manual_keywords = [
        'instruction', 'instructions',
        'manual', 'documentation',
        'warning', 'warnings', 'caution', 'notice',
        'marking', 'label', 'labeling', 'labelling',
        'pictogram', 'symbol', 'icon',
        'user information', 'information to user',
        'owner manual', 'operating instructions',
        'maintenance instructions',
        'assembly instructions', 'installation instructions',
    ]

    for keyword in manual_keywords:
        if keyword in text_lower:
            return 'y'

    return 'n'


def validate_requirement(req: Dict) -> Tuple[bool, str]:
    """
    Tag and validate requirements - INCLUDE everything except obvious junk.

    NEW PHILOSOPHY: Tag clauses by type instead of rejecting them.
    Definitions, test methods, preambles are VALID - just tag them for filtering later.

    Returns:
        (is_valid, rejection_reason)
    """
    clause = req.get('clause', '').strip()
    text = req.get('text', '').strip()

    # Clean Comments field
    req = clean_comments_field(req)

    # REJECT only obvious junk

    # 1. N/A placeholders - still reject (no value)
    is_na, reason = is_na_placeholder(text)
    if is_na:
        return False, reason

    # 2. Minimum substance check - still reject (garbage)
    if not has_minimum_substance(text):
        return False, f"Too short or lacks substance ({len(text)} chars)"

    # TAG instead of reject - keep all clauses but classify them

    # 3. Preamble → TAG as "Preamble"
    is_pream, reason = is_preamble(text)
    if is_pream:
        req['Clause_Type'] = 'Preamble'
    # 4. Pure definitions → TAG as "Definition"
    elif is_pure_definition(clause, text)[0]:
        req['Clause_Type'] = 'Definition'
    # 5. Pure test methodology → TAG as "Test_Methodology"
    elif is_pure_test_methodology(text)[0]:
        req['Clause_Type'] = 'Test_Methodology'
    # 6. Default → TAG as "Requirement"
    else:
        req['Clause_Type'] = 'Requirement'

    # 7. Populate parent section (always)
    parent = parse_parent_section(clause)
    if parent:
        req['Parent Section'] = parent
    elif 'Parent Section' not in req:
        req['Parent Section'] = 'Unknown'

    # 8. Add classification tags (mandate level, safety, manual)
    req['Mandate_Level'] = classify_mandate_level(text)
    req['Safety_Flag'] = classify_safety_flag(text)
    req['Manual_Flag'] = classify_manual_flag(text)

    # INCLUDE IT with appropriate tag
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
        # Preserve all original fields + add new classification columns if not present
        output_fieldnames = list(fieldnames) if fieldnames else ['clause', 'text']
        new_columns = ['Parent Section', 'Clause_Type', 'Mandate_Level', 'Safety_Flag', 'Manual_Flag']
        for col in new_columns:
            if col not in output_fieldnames:
                output_fieldnames.append(col)

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
