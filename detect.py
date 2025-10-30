"""
Detection logic for manual sections and manual-related clauses.
"""
import re
from typing import Dict, List, Tuple


# Patterns for detecting manual/instructions sections (Pass A)
MANUAL_SECTION_PATTERNS = [
    r'instructions?\s+for\s+use',
    r'information\s+for\s+use',
    r'content\s+of\s+the\s+instructions?',
    r'requirements?\s+for\s+manuals?',
    r'marking\s+and\s+instructions?',
    r'user\s+information',
    r'accompanying\s+documents?',
    r'documentation',
    r'requirements?\s+for\s+the\s+requirements?',  # Meta requirements
    r'^\d+\.\d+\.\d+\.?\d*\s+instructions?',  # e.g., 1.7.4 instructions
    r'operating\s+instructions?',
    r'safety\s+instructions?',
]

# Patterns for detecting manual-related clauses anywhere (Pass B)
CLAUSE_KEYWORD_PATTERNS = [
    r'shall\s+accompany',
    r'shall\s+be\s+supplied\s+with',
    r'shall\s+be\s+included\s+with\s+the\s+product',
    r'shall\s+be\s+included\s+in\s+the\s+instruction\s+manual',
    r'shall\s+be\s+provided\s+to\s+the\s+user',
    r'instructions?\s+shall',
    r'information\s+shall\s+be\s+made\s+available',
    r'user\s+instructions?',
    r'operating\s+instructions?',
    r'safety\s+instructions?',
    r'\bWARNING\b',
    r'\bCAUTION\b',
    r'\bDANGER\b',
    r'marking\s+and\s+instructions?',
    r'instruction\s+manual',
    r'included\s+with\s+the\s+packaged\s+unit',
]

# Patterns for extracting clause numbers
CLAUSE_NUMBER_PATTERNS = [
    r'^\d+(\.\d+)*',           # 1.7.4.1
    r'^[A-Z]\.\d+(\.\d+)*',    # A.2.3
    r'^[IVX]+\.\d+',           # VII.4
    r'^\d+\.\d+\([a-z]\)',     # 1.2(a)
]


def detect_manual_sections(blocks: List[Dict]) -> List[Dict]:
    """
    Pass A: Detect sections dedicated to instructions/manuals.

    Args:
        blocks: List of text blocks from extract_text_blocks()

    Returns:
        List of detected sections with keys:
        - 'start_line': line number where section starts
        - 'end_line': line number where section ends (or None if until end)
        - 'heading': the heading text
        - 'clause_number': extracted clause number if found
        - 'content': all text in this section
    """
    sections = []

    for i, block in enumerate(blocks):
        if not block['maybe_heading']:
            continue

        heading_text = block['raw'].strip()
        heading_lower = heading_text.lower()

        # Check if this heading matches any manual section pattern
        is_manual_section = False
        for pattern in MANUAL_SECTION_PATTERNS:
            if re.search(pattern, heading_lower, re.IGNORECASE):
                is_manual_section = True
                break

        if not is_manual_section:
            continue

        # Extract clause number from heading
        clause_number = extract_clause_number(heading_text)

        # Find the end of this section (next heading of same or higher level)
        section_content = [heading_text]
        end_line = None

        for j in range(i + 1, len(blocks)):
            if blocks[j]['maybe_heading']:
                # Check if this is a heading at same or higher level
                # For simplicity, we'll treat any heading as a section boundary
                end_line = blocks[j]['lineno'] - 1
                break
            else:
                section_content.append(blocks[j]['raw'])

        if end_line is None:
            end_line = blocks[-1]['lineno']

        sections.append({
            'start_line': block['lineno'],
            'end_line': end_line,
            'heading': heading_text,
            'clause_number': clause_number,
            'content': '\n'.join(section_content)
        })

    return sections


def detect_manual_clauses(blocks: List[Dict], manual_sections: List[Dict]) -> List[Dict]:
    """
    Pass B: Find clauses anywhere in the document that mention manual/instructions.

    Args:
        blocks: List of text blocks
        manual_sections: List of already-detected manual sections (to avoid duplicates)

    Returns:
        List of detected clauses with keys:
        - 'text': the clause text
        - 'line_number': line number
        - 'matched_pattern': which pattern matched
        - 'clause_number': extracted clause number if found
        - 'context': surrounding lines for context
    """
    clauses = []

    # Build set of line numbers already covered by manual sections
    covered_lines = set()
    for section in manual_sections:
        for line_no in range(section['start_line'], section['end_line'] + 1):
            covered_lines.add(line_no)

    # Scan all blocks for keyword patterns
    for i, block in enumerate(blocks):
        line_no = block['lineno']

        # Skip if already in a manual section
        if line_no in covered_lines:
            continue

        text = block['raw'].strip()
        if not text or len(text) < 10:  # Skip very short lines
            continue

        # Check against all clause keyword patterns
        matched_patterns = []
        for pattern in CLAUSE_KEYWORD_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matched_patterns.append(pattern)

        if not matched_patterns:
            continue

        # Extract clause number if present
        clause_number = extract_clause_number(text)

        # Get context (2 lines before and after)
        context_lines = []
        for j in range(max(0, i - 2), min(len(blocks), i + 3)):
            context_lines.append(blocks[j]['raw'])

        clauses.append({
            'text': text,
            'line_number': line_no,
            'matched_patterns': matched_patterns,
            'clause_number': clause_number,
            'context': '\n'.join(context_lines)
        })

    return clauses


def extract_clause_number(text: str) -> str:
    """
    Extract clause number from text using common patterns.

    Args:
        text: Text that may contain a clause number

    Returns:
        Extracted clause number or empty string
    """
    text_stripped = text.strip()

    for pattern in CLAUSE_NUMBER_PATTERNS:
        match = re.match(pattern, text_stripped)
        if match:
            return match.group(0)

    return ''


def extract_all_paragraphs(blocks: List[Dict]) -> List[Dict]:
    """
    Group blocks into paragraphs (separated by blank lines).

    Args:
        blocks: List of text blocks

    Returns:
        List of paragraphs with keys:
        - 'text': paragraph text
        - 'start_line': starting line number
        - 'end_line': ending line number
        - 'is_heading': whether this paragraph is a heading
    """
    paragraphs = []
    current_paragraph = []
    start_line = None

    for block in blocks:
        line = block['raw'].strip()

        if not line:  # Blank line - end current paragraph
            if current_paragraph:
                paragraphs.append({
                    'text': '\n'.join(current_paragraph),
                    'start_line': start_line,
                    'end_line': block['lineno'] - 1,
                    'is_heading': blocks[start_line - 1]['maybe_heading'] if start_line else False
                })
                current_paragraph = []
                start_line = None
        else:
            if not current_paragraph:
                start_line = block['lineno']
            current_paragraph.append(line)

    # Add final paragraph if exists
    if current_paragraph and start_line:
        paragraphs.append({
            'text': '\n'.join(current_paragraph),
            'start_line': start_line,
            'end_line': blocks[-1]['lineno'],
            'is_heading': blocks[start_line - 1]['maybe_heading'] if start_line else False
        })

    return paragraphs


def combine_detections(manual_sections: List[Dict], manual_clauses: List[Dict]) -> List[Dict]:
    """
    Combine Pass A (sections) and Pass B (clauses) results, deduplicating.

    Args:
        manual_sections: Detected manual sections
        manual_clauses: Detected manual clauses

    Returns:
        Combined list of detected items with keys:
        - 'text': the text content
        - 'source': 'section' or 'clause'
        - 'line_number': line number or range
        - 'clause_number': extracted clause number
        - 'heading': heading if from section
        - 'matched_patterns': patterns if from clause search
        - 'context': additional context
    """
    combined = []

    # Add all manual sections
    for section in manual_sections:
        combined.append({
            'text': section['content'],
            'source': 'section',
            'line_number': f"{section['start_line']}-{section['end_line']}",
            'clause_number': section['clause_number'],
            'heading': section['heading'],
            'matched_patterns': [],
            'context': section['content']
        })

    # Add all manual clauses
    for clause in manual_clauses:
        combined.append({
            'text': clause['text'],
            'source': 'clause',
            'line_number': str(clause['line_number']),
            'clause_number': clause['clause_number'],
            'heading': '',
            'matched_patterns': clause['matched_patterns'],
            'context': clause['context']
        })

    return combined
