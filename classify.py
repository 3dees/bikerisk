"""
Classification logic to populate the 8-column requirement schema.
"""
import re
from typing import Dict, List


# Hard requirement phrases (must be included)
HARD_REQUIREMENT_PATTERNS = [
    r'shall\s+accompany\s+(the\s+)?product',
    r'shall\s+be\s+supplied\s+with\s+(the\s+)?product',
    r'shall\s+be\s+included\s+with\s+(the\s+)?product',
    r'shall\s+be\s+included\s+in\s+the\s+instruction\s+manual',
    r'instructions?\s+shall\s+be\s+provided\s+to\s+the\s+user',
    r'shall\s+be\s+provided\s+with',
]

# Soft requirement phrases (may be ambiguous)
SOFT_REQUIREMENT_PATTERNS = [
    r'shall\s+be\s+made\s+available',
    r'information\s+for\s+the\s+user',
    r'may\s+be\s+provided',
    r'should\s+be\s+available',
]

# Warning/caution/danger tokens (always required in print)
WARNING_PATTERNS = [
    r'\bWARNING\b',
    r'\bCAUTION\b',
    r'\bDANGER\b',
    r'\bNOTE\b',
]

# Exclusion patterns for warnings (when it's NOT a safety warning)
WARNING_EXCLUSION_PATTERNS = [
    r'warning\s+device',
    r'warning\s+signal',
    r'warning\s+light',
    r'warning\s+bell',
    r'audible\s+warning',
    r'visual\s+warning',
    r'warning\s+system',
]

# Exclusion patterns for manual (when it's NOT about instruction manual)
MANUAL_EXCLUSION_PATTERNS = [
    r'manual\s+(or|and|vs)\s+automatic',
    r'manual\s+operation',
    r'manual\s+control',
    r'manual\s+transmission',
    r'manual\s+adjustment',
    r'manually\s+operated',
]

# Scope detection patterns
SCOPE_PATTERNS = {
    'battery': [
        r'\bbattery\b',
        r'\bbattery\s+pack\b',
        r'\baccumulator\b',
        r'\bcell\b',
    ],
    'charger': [
        r'\bcharger\b',
        r'\bcharging\s+equipment\b',
        r'\bcharging\b',
    ],
    'ebike': [
        r'\be-bike\b',
        r'\bebike\b',
        r'\bpedelec\b',
        r'\belectric\s+bicycle\b',
        r'\belectric\s+bike\b',
    ],
    'bicycle': [
        r'\bbicycle\b',
        r'\bbike\b',
    ],
}

# Formatting requirement patterns
FORMATTING_PATTERNS = [
    r'shall\s+be\s+in\s+capital\s+letters',
    r'shall\s+be\s+clearly\s+legible',
    r'shall\s+be\s+in\s+the\s+official\s+language',
    r'shall\s+bear\s+the\s+words\s+[\'"]Original\s+instructions[\'"]',
    r'shall\s+be\s+indelible',
    r'shall\s+be\s+permanent',
    r'minimum\s+font\s+size',
    r'shall\s+be\s+visible',
]

# Print requirement patterns (explicit mentions of print/paper)
PRINT_REQUIREMENT_PATTERNS = [
    r'shall\s+be\s+printed',
    r'printed\s+manual',
    r'paper\s+copy',
    r'physical\s+copy',
    r'in\s+print',
]


def classify_detected_items(
    detected_items: List[Dict],
    standard_name: str = "Unknown Standard"
) -> List[Dict]:
    """
    Classify detected items into the 8-column schema.

    Args:
        detected_items: Items from combine_detections()
        standard_name: Name of the standard/regulation being processed

    Returns:
        List of classified rows with schema:
        - Requirement (Clause)
        - Standard/ Regulation
        - Clause
        - Must be included with product?
        - Requirement Scope
        - Formatting Requirement(s)?
        - Required in Print?
        - Comments
    """
    classified = []

    for item in detected_items:
        row = _classify_single_item(item, standard_name)
        if row:
            classified.append(row)

    return classified


def _classify_single_item(item: Dict, standard_name: str) -> Dict:
    """
    Classify a single detected item.

    Args:
        item: Single detected item
        standard_name: Name of standard

    Returns:
        Classified row dict or None if should be ignored
    """
    text = item['text']
    text_lower = text.lower()

    # Determine must_be_included and comments
    must_be_included = 'N'
    comments = []
    confidence = 'MEDIUM'

    # Rule 1: Inside manual/instructions section
    if item['source'] == 'section':
        must_be_included = 'Y'
        comments.append('inside instructions section')
        confidence = 'HIGH'

    # Rule 2: Hard requirement phrase
    elif _has_hard_requirement(text_lower):
        must_be_included = 'Y'
        comments.append('explicit accompany/supply')
        confidence = 'HIGH'

    # Rule 3: Warning/caution/danger token
    elif _has_warning_token(text):
        must_be_included = 'Y'
        comments.append('warning token')
        confidence = 'HIGH'

    # Rule 4: Soft requirement phrase
    elif _has_soft_requirement(text_lower):
        must_be_included = 'Ambiguous'
        comments.append('made available / may → review')
        confidence = 'MEDIUM'

    # Rule 5: Found by keyword but no strong signal
    elif item['source'] == 'clause' and item['matched_patterns']:
        must_be_included = 'N'
        comments.append('keyword match only')
        confidence = 'LOW'

    else:
        # No clear classification - skip
        return None

    # Determine scope
    scope = _detect_scope(text_lower)

    # Determine formatting requirements
    formatting = _detect_formatting(text)

    # Determine print requirement
    required_in_print = _determine_print_requirement(text_lower, must_be_included, item)

    # Extract clause number
    clause = item['clause_number'] if item['clause_number'] else item.get('heading', '')

    return {
        'Description': text.strip(),
        'Standard/Reg': standard_name,
        'Clause/Requirement': clause,
        'Requirement scope': scope,
        'Formatting required?': formatting,
        'Required in Print?': required_in_print,
        'Comments': '; '.join(comments) if comments else '',
        '_confidence': confidence,  # Internal field
        '_line_number': item['line_number'],  # Internal field for debugging
    }


def _has_hard_requirement(text_lower: str) -> bool:
    """Check if text contains hard requirement phrases."""
    for pattern in HARD_REQUIREMENT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def _has_soft_requirement(text_lower: str) -> bool:
    """Check if text contains soft requirement phrases."""
    for pattern in SOFT_REQUIREMENT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def _has_warning_token(text: str) -> bool:
    """
    Check if text contains warning/caution/danger tokens.

    Excludes cases where it's referring to warning devices, not safety warnings.
    """
    text_lower = text.lower()

    # First check if any warning pattern matches
    has_warning = False
    for pattern in WARNING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            has_warning = True
            break

    if not has_warning:
        return False

    # Now check if it's an exclusion case (warning device, not safety warning)
    for exclusion in WARNING_EXCLUSION_PATTERNS:
        if re.search(exclusion, text_lower):
            return False  # It's a warning device, not a safety warning

    return True  # It's a real safety warning


def _detect_scope(text_lower: str) -> str:
    """Detect requirement scope (battery, charger, ebike, bicycle)."""
    scopes = []

    for scope_name, patterns in SCOPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                scopes.append(scope_name)
                break  # Only add each scope once

    # Return comma-separated unique scopes
    return ', '.join(sorted(set(scopes))) if scopes else ''


def _detect_formatting(text: str) -> str:
    """Detect formatting requirements."""
    formatting_reqs = []

    for pattern in FORMATTING_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Extract the matched phrase
            formatting_reqs.append(match.group(0))

    return '; '.join(formatting_reqs) if formatting_reqs else 'N/A'


def _determine_print_requirement(text_lower: str, must_be_included: str, item: Dict) -> str:
    """
    Determine if requirement must be in print.

    Rules:
    1. Explicit print mention → 'y'
    2. Warning/caution/danger → 'y'
    3. Must be included from instructions section → 'y'
    4. Otherwise → 'n'
    """
    # Rule 1: Explicit print requirement
    for pattern in PRINT_REQUIREMENT_PATTERNS:
        if re.search(pattern, text_lower):
            return 'y'

    # Rule 2: Warning tokens
    if _has_warning_token(item['text']):
        return 'y'

    # Rule 3: From instructions section and must be included
    if item['source'] == 'section' and must_be_included == 'Y':
        return 'y'

    # Default
    return 'n'


def rows_to_csv_dicts(rows: List[Dict]) -> List[Dict]:
    """
    Convert classified rows to CSV-friendly format (remove internal fields).

    Args:
        rows: Classified rows with internal fields

    Returns:
        List of dicts with 12 schema columns (including requirement_id, Parent Section, and Sub-section)
    """
    csv_rows = []

    for row in rows:
        csv_rows.append({
            'Requirement ID': row.get('requirement_id', ''),
            'Description': row.get('Description', row.get('text', '')),
            'Standard/Reg': row.get('Standard/Reg', ''),
            'Clause/Requirement': row.get('Clause/Requirement', ''),
            'Requirement scope': row.get('Requirement scope', ''),
            'Formatting required?': row.get('Formatting required?', 'N/A'),
            'Required in Print?': row.get('Required in Print?', 'n'),
            'Parent Section': row.get('Parent Section', 'Unknown'),
            'Sub-section': row.get('Sub-section', 'N/A'),
            'Comments': row.get('Comments', ''),
            'Contains Image?': row.get('Contains Image?', 'N'),
            'Safety Notice Type': row.get('Safety Notice Type', 'None'),
            # NEW COLUMNS from validate.py tagging
            'Clause_Type': row.get('Clause_Type', 'Requirement'),
            'Mandate_Level': row.get('Mandate_Level', 'Informative'),
            'Safety_Flag': row.get('Safety_Flag', 'n'),
            'Manual_Flag': row.get('Manual_Flag', 'n'),
        })

    return csv_rows
