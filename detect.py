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

# Patterns for extracting clause numbers
CLAUSE_NUMBER_PATTERNS = [
    r'^\d+(\.\d+)*',           # 1.7.4.1
    r'^[A-Z]\.\d+(\.\d+)*',    # A.2.3
    r'^[IVX]+\.\d+',           # VII.4
    r'^\d+\.\d+\([a-z]\)',     # 1.2(a)
]


def detect_manual_sections(blocks: List[Dict], custom_section_names: List[str] = None) -> List[Dict]:
    """
    Pass A: Detect sections dedicated to instructions/manuals.

    Args:
        blocks: List of text blocks from extract_text_blocks()
        custom_section_names: Optional list of custom section names to search for (e.g., ["Instruction for use"])

    Returns:
        List of detected sections with keys:
        - 'start_line': line number where section starts
        - 'end_line': line number where section ends (or None if until end)
        - 'heading': the heading text
        - 'clause_number': extracted clause number if found
        - 'content': all text in this section
    """
    sections = []

    # Combine default patterns with custom names
    search_patterns = MANUAL_SECTION_PATTERNS.copy()
    if custom_section_names:
        for name in custom_section_names:
            # Escape special regex chars and add to patterns
            escaped_name = re.escape(name.lower())
            search_patterns.append(escaped_name)

    for i, block in enumerate(blocks):
        if not block['maybe_heading']:
            continue

        heading_text = block['raw'].strip()
        heading_lower = heading_text.lower()

        # Skip if this looks like a TOC entry (has dots followed by page numbers)
        if re.search(r'\.{3,}\s*\d+\s*$', heading_text):
            continue

        # Check if this heading matches any manual section pattern
        is_manual_section = False
        for pattern in search_patterns:
            if re.search(pattern, heading_lower, re.IGNORECASE):
                is_manual_section = True
                break

        if not is_manual_section:
            continue

        # Extract clause number from heading
        clause_number = extract_clause_number(heading_text)

        # Find the end of this section (next heading of same or higher level)
        section_content = []
        section_content.append(heading_text)
        end_line = None

        for j in range(i + 1, len(blocks)):
            line_text = blocks[j]['raw'].strip()

            # Skip TOC entries, page numbers, and other noise
            if not line_text:
                continue
            if re.search(r'\.{3,}\s*\d+\s*$', line_text):  # TOC dots and page numbers
                continue
            if re.match(r'^\d+\s*$', line_text):  # Standalone page numbers
                continue
            if re.match(r'^EN\s+\d+:', line_text):  # Standard references like "EN 15194:2017"
                continue

            # Check if we hit the next major heading (indicates section end)
            if blocks[j]['maybe_heading']:
                next_clause = extract_clause_number(line_text)
                # Only end section if we hit a heading with a different top-level clause number
                if next_clause and clause_number:
                    # Extract top-level numbers
                    current_top = clause_number.split('.')[0] if '.' in clause_number else clause_number
                    next_top = next_clause.split('.')[0] if '.' in next_clause else next_clause
                    if current_top != next_top and not next_clause.startswith(clause_number):
                        end_line = blocks[j]['lineno'] - 1
                        break

            section_content.append(line_text)

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

        # Check for exclusions
        text_lower = text.lower()

        # Skip if it's a warning device reference (not a safety warning)
        is_warning_device = False
        if any(re.search(r'\b(WARNING|CAUTION|DANGER)\b', text, re.IGNORECASE) for _ in [1]):
            for exclusion in WARNING_EXCLUSION_PATTERNS:
                if re.search(exclusion, text_lower):
                    is_warning_device = True
                    break

        # Skip if it's about manual operation (not instruction manual)
        is_manual_operation = False
        if 'manual' in text_lower:
            for exclusion in MANUAL_EXCLUSION_PATTERNS:
                if re.search(exclusion, text_lower):
                    is_manual_operation = True
                    break

        # Skip this clause if it matches an exclusion
        if is_warning_device or is_manual_operation:
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


def split_lettered_items(section_content: str, parent_clause: str, heading: str) -> List[Dict]:
    """
    Split section content into individual lettered sub-items (a), b), c), etc.).

    Args:
        section_content: Full section text
        parent_clause: Parent clause number (e.g., "6")
        heading: Section heading

    Returns:
        List of sub-items, each with text and sub-clause number
    """
    # Pattern to match lettered items: a), b), aa), bb), etc.
    letter_pattern = r'^([a-z]+)\)\s+'

    items = []
    lines = section_content.split('\n')

    # Pre-filter garbage lines
    lines = [line for line in lines if not _is_garbage_line(line.strip())]

    current_item = []
    current_letter = None
    intro_text = []  # Text before first lettered item

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            continue

        # Check if this line starts with a letter pattern
        match = re.match(letter_pattern, line_stripped)

        if match:
            # Save previous item if exists
            if current_item and current_letter:
                item_text = '\n'.join(current_item).strip()
                items.append({
                    'text': item_text,
                    'letter': current_letter,
                    'clause_number': f"{current_letter})"  # Just the letter with parenthesis
                })

            # Start new item
            current_letter = match.group(1)
            # Remove the letter prefix from the line
            item_text = re.sub(letter_pattern, '', line_stripped)
            current_item = [item_text]

        else:
            # Continue current item or intro text
            if current_letter:
                current_item.append(line_stripped)
            else:
                intro_text.append(line_stripped)

    # Save final item
    if current_item and current_letter:
        item_text = '\n'.join(current_item).strip()
        items.append({
            'text': item_text,
            'letter': current_letter,
            'clause_number': f"{current_letter})"  # Just the letter with parenthesis
        })

    # If we found lettered items, also include intro text as item 0
    if items and intro_text:
        intro_clean = '\n'.join(intro_text).strip()
        # Remove the heading from intro
        if intro_clean.startswith(heading):
            intro_clean = intro_clean[len(heading):].strip()

        if intro_clean and len(intro_clean) > 50:  # Only include if substantial
            items.insert(0, {
                'text': intro_clean,
                'letter': 'intro',
                'clause_number': parent_clause  # Parent clause gets the main number
            })

    return items


def split_numbered_subsections(section_content: str, parent_clause: str, heading: str) -> List[Dict]:
    """
    Split section content into individual numbered sub-sections (7.1, 7.6, 7.12, etc.).

    Args:
        section_content: Full section text
        parent_clause: Parent clause number (e.g., "7")
        heading: Section heading

    Returns:
        List of sub-items, each with text and sub-clause number
    """
    # Pattern to match numbered subsections: 7.1, 7.12, 7.12.1, etc.
    # Must start at beginning of line and match parent clause
    # May be followed by optional text like "Addition:" or "Replacement:"
    if not parent_clause:
        return []

    # DEBUG: Print what we're looking for
    print(f"[DEBUG] split_numbered_subsections called:")
    print(f"  parent_clause: '{parent_clause}'")
    print(f"  heading: '{heading[:50]}...' (truncated)")
    print(f"  content length: {len(section_content)} chars")

    # Try strict pattern first (must match parent clause)
    # Match patterns like "7.1 Addition:" or "7.12 " or "7.12.1"
    # Be flexible with spacing and optional keywords
    strict_pattern = rf'^({re.escape(parent_clause)}\.\d+(?:\.\d+)*)\s*(?:Addition:|Replacement:|Amendment:|Modification:)?:?\s*'

    # Fallback: if strict doesn't work, try any numbered subsection
    # This matches patterns like "7.1 ", "19.2.3 ", etc.
    loose_pattern = rf'^(\d+\.\d+(?:\.\d+)*)\s*(?:Addition:|Replacement:|Amendment:|Modification:)?:?\s*'

    print(f"  strict_pattern: {strict_pattern}")
    print(f"  loose_pattern: {loose_pattern}")

    items = []
    lines = section_content.split('\n')

    # Pre-filter garbage lines and clean up
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not _is_garbage_line(stripped):
            clean_lines.append(stripped)

    print(f"  clean_lines count: {len(clean_lines)} (after filtering {len(lines) - len(clean_lines)} garbage lines)")

    # Show first few clean lines
    print(f"  First 5 clean lines:")
    for i, line in enumerate(clean_lines[:5]):
        print(f"    [{i}]: {line[:80]}...")

    current_item = []
    current_number = None
    intro_text = []  # Text before first numbered subsection
    matches_found = 0

    for line_stripped in clean_lines:
        # Try strict pattern first
        match = re.match(strict_pattern, line_stripped)
        used_pattern = strict_pattern
        pattern_type = "strict"
        if not match:
            # Fallback to loose pattern
            match = re.match(loose_pattern, line_stripped)
            used_pattern = loose_pattern
            pattern_type = "loose"

        if match:
            matches_found += 1
            print(f"  [MATCH {matches_found}] {pattern_type} pattern matched: {match.group(1)} in line: {line_stripped[:60]}...")

            # Save previous item if exists
            if current_item and current_number:
                item_text = '\n'.join(current_item).strip()
                items.append({
                    'text': item_text,
                    'number': current_number,
                    'clause_number': current_number
                })

            # Start new item
            current_number = match.group(1)
            # Remove the number prefix from the line
            item_text = re.sub(used_pattern, '', line_stripped)
            current_item = [item_text]

        else:
            # Continue current item or intro text
            if current_number:
                current_item.append(line_stripped)
            else:
                intro_text.append(line_stripped)

    # Save final item
    if current_item and current_number:
        item_text = '\n'.join(current_item).strip()
        items.append({
            'text': item_text,
            'number': current_number,
            'clause_number': current_number
        })

    # If we found numbered subsections, also include intro text as item 0
    if items and intro_text:
        intro_clean = '\n'.join(intro_text).strip()
        # Remove the heading from intro
        if intro_clean.startswith(heading):
            intro_clean = intro_clean[len(heading):].strip()

        if intro_clean and len(intro_clean) > 50:  # Only include if substantial
            items.insert(0, {
                'text': intro_clean,
                'number': 'intro',
                'clause_number': parent_clause  # Parent clause gets the main number
            })

    print(f"  [RESULT] Found {len(items)} subsections")
    if items:
        for i, item in enumerate(items[:3]):
            print(f"    Item {i}: clause={item['clause_number']}, text={item['text'][:50]}...")
    print()

    return items


def _is_garbage_line(line: str) -> bool:
    """
    Detect garbage header/footer lines (watermarks, page headers, etc.).

    Common patterns:
    - Reversed text like ".oN redrO" or "redrO rof"
    - Copyright notices with weird spacing
    - Repeating headers/footers
    """
    if not line:
        return False

    # Pattern 1: Reversed/scrambled text (has dots and backwards words)
    if re.search(r'\.(oN|redrO|sresu|thgirypoc|ot|tcejbus|era|sdradnatS)', line):
        return True

    # Pattern 2: Common reversed words (even without dots) - comprehensive list
    # Check for individual reversed words first (these appear on separate lines)
    individual_reversed = [
        r'\bredrO\b', r'\bsresu\b', r'\bthgirypoc\b', r'\btcejbus\b',
        r'\bsdradnatS\b', r'\bnoitasidradnatS\b', r'\bertneC\b',
        r'\bnainotsE\b', r'\bsgnoleb\b', r'\betubirtsid\b', r'\becudorper\b',
        r'\bkerT\b', r'\bcinortcele\b', r'\btnemucod\b', r'\besiU\b', r'\besU\b',
        # Individual short reversed words (common in garbage)
        r'\brof\b', r'\becnecil\b', r'\bresu\b', r'\bitluM\b',
        r'\bot\b', r'\bera\b', r'\bdna\b', r'\beht\b',
        r'\bthgir\b', r'\bsiht\b', r'\bfo\b'
    ]
    for word in individual_reversed:
        if re.search(word, line):
            return True

    # Check for multi-word reversed phrases
    if re.search(r'\brof\b.*\becnecil\b', line) or re.search(r'\bitluM\b.*\bresu\b', line):
        return True

    # Pattern 3: Specific garbage patterns from this document
    if re.search(r'(rof ecnecil|resu itluM|ot tcejbus|era sdradnatS|thgir eht)', line):
        return True

    # Pattern 4: Lines with excessive punctuation relative to words
    punct_count = len(re.findall(r'[.,;:]', line))
    word_count = len(re.findall(r'\w+', line))
    if word_count > 0 and punct_count / word_count > 0.5:
        return True

    # Pattern 5: Very long lines with no spaces (garbled text)
    if len(line) > 100 and ' ' not in line:
        return True

    # Pattern 6: Page number patterns like "– 13 – EVS-EN IEC" or "– 14 –" or "– 23 –"
    if re.search(r'–\s+\d+\s+–\s*(EVS-EN|IEC|EN\s+IEC)?', line):
        return True

    # Pattern 7: Lines that are mostly punctuation/numbers (like ".2202.50.81,779474.oN")
    if re.match(r'^[\d.,\s]+[a-zA-Z]{2,}$', line):
        return True

    # Pattern 8: Lines that start with period followed by numbers (reversed dates like ".2202.50.81")
    if re.match(r'^\.\d{4}\.\d{2}\.\d{2}', line):
        return True

    # Pattern 9: Lines that start with comma followed by numbers (reversed order numbers like ",779474")
    if re.match(r'^,\d+', line):
        return True

    # Pattern 10: Very short lines with just reversed abbreviations (like ".oN" or ",SVE")
    if re.match(r'^[.,]\w{2,5}$', line):
        return True

    return False


def _filter_garbage_from_text(text: str) -> str:
    """
    Remove garbage lines from text content.

    Args:
        text: Multi-line text content

    Returns:
        Filtered text with garbage lines removed
    """
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not _is_garbage_line(stripped):
            clean_lines.append(line)
    return '\n'.join(clean_lines)


def combine_detections(manual_sections: List[Dict], manual_clauses: List[Dict]) -> List[Dict]:
    """
    Combine Pass A (sections) and Pass B (clauses) results, deduplicating.

    For sections with lettered sub-items, split them into individual requirements.

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

    # Add all manual sections (with sub-item splitting)
    for section in manual_sections:
        # Try numbered subsections first (7.1, 7.6, 7.12, etc.)
        sub_items = split_numbered_subsections(
            section['content'],
            section['clause_number'],
            section['heading']
        )

        # If no numbered subsections, try lettered items (a), b), c), etc.)
        if not sub_items:
            sub_items = split_lettered_items(
                section['content'],
                section['clause_number'],
                section['heading']
            )

        if sub_items:
            # Add each sub-item as a separate detection
            for sub_item in sub_items:
                combined.append({
                    'text': sub_item['text'],
                    'source': 'section',
                    'line_number': f"{section['start_line']}-{section['end_line']}",
                    'clause_number': sub_item['clause_number'],
                    'heading': section['heading'],
                    'matched_patterns': [],
                    'context': section['content']
                })
        else:
            # No sub-items found, filter garbage and add entire section
            filtered_content = _filter_garbage_from_text(section['content'])
            combined.append({
                'text': filtered_content,
                'source': 'section',
                'line_number': f"{section['start_line']}-{section['end_line']}",
                'clause_number': section['clause_number'],
                'heading': section['heading'],
                'matched_patterns': [],
                'context': filtered_content
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
