"""
Text extraction from PDF files with fallback mechanisms.
"""
import io
from typing import Dict, List, Optional
import pdfplumber
import pypdf
from extract_ocr import extract_with_ocr


def extract_text_blocks(text: str) -> List[Dict]:
    """
    Parse extracted text into structured blocks with line numbers and heading detection.

    Args:
        text: Raw extracted text

    Returns:
        List of dicts with keys: 'raw', 'lineno', 'maybe_heading'
    """
    blocks = []
    lines = text.split('\n')

    for lineno, line in enumerate(lines, start=1):
        # Detect potential headings by looking for common patterns
        maybe_heading = _is_likely_heading(line)

        blocks.append({
            'raw': line,
            'lineno': lineno,
            'maybe_heading': maybe_heading
        })

    return blocks


def _is_likely_heading(line: str) -> bool:
    """
    Detect if a line is likely a heading based on numbering patterns.

    Patterns:
    - 1.7.4.1 General principles...
    - A.2.3 Contents...
    - VII.4.a Instructions...
    - 1.2(a) User information...
    """
    import re

    line_stripped = line.strip()
    if not line_stripped:
        return False

    # Common heading patterns
    patterns = [
        r'^\d+(\.\d+)+\s+',         # 1.7.4.1 ...
        r'^[A-Z]\.\d+\s+',           # A.2.3 ...
        r'^[IVX]+\.\d+\s+',          # VII.4 ...
        r'^\d+\.\d+\([a-z]\)\s+',    # 1.2(a) ...
        r'^\d+\s+[A-Z]',             # 7 General...
    ]

    for pattern in patterns:
        if re.match(pattern, line_stripped):
            return True

    # Also consider lines that are all caps and short (< 80 chars) as headings
    if line_stripped.isupper() and len(line_stripped) < 80 and len(line_stripped) > 3:
        return True

    return False


def extract_from_pdf(file_bytes: bytes, filename: str) -> Dict:
    """
    Extract text from PDF using pdfplumber, falling back to pypdf if needed.

    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename for logging

    Returns:
        Dict with keys:
        - 'raw_text': extracted text
        - 'blocks': structured text blocks
        - 'method': extraction method used ('pdfplumber' or 'pypdf')
        - 'success': bool
        - 'error': error message if failed
        - 'confidence': extraction confidence ('high', 'medium', 'low')
    """
    # Try pdfplumber first
    try:
        text, method = _extract_with_pdfplumber(file_bytes)
        if text and len(text.strip()) > 100:  # Meaningful content
            blocks = extract_text_blocks(text)
            return {
                'raw_text': text,
                'blocks': blocks,
                'method': 'pdfplumber',
                'success': True,
                'error': None,
                'confidence': 'high'
            }
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    # Fallback to pypdf
    try:
        text, method = _extract_with_pypdf(file_bytes)
        if text and len(text.strip()) > 100:
            blocks = extract_text_blocks(text)
            return {
                'raw_text': text,
                'blocks': blocks,
                'method': 'pypdf',
                'success': True,
                'error': None,
                'confidence': 'medium'
            }
    except Exception as e:
        return {
            'raw_text': '',
            'blocks': [],
            'method': 'none',
            'success': False,
            'error': f'Both extraction methods failed. pdfplumber and pypdf errors. Last error: {str(e)}',
            'confidence': 'low'
        }

    # If we got here, extraction produced no meaningful content
    # Try OCR as last resort for image-based PDFs
    print(f"[EXTRACTION] Text extraction failed for {filename}, attempting OCR...")

    try:
        ocr_result = extract_with_ocr(file_bytes, filename)

        if ocr_result['success']:
            # OCR succeeded, add blocks
            blocks = extract_text_blocks(ocr_result['raw_text'])
            ocr_result['blocks'] = blocks
            print(f"[EXTRACTION] OCR successful using {ocr_result['method']}")
            return ocr_result
        else:
            print(f"[EXTRACTION] OCR failed: {ocr_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"[EXTRACTION] OCR exception: {str(e)}")

    # All methods failed
    return {
        'raw_text': text if 'text' in locals() else '',
        'blocks': [],
        'method': 'all_failed',
        'success': False,
        'error': 'All extraction methods failed (pdfplumber, pypdf, and OCR). PDF may be corrupted or heavily encrypted.',
        'confidence': 'low'
    }


def _extract_with_pdfplumber(file_bytes: bytes) -> tuple[str, str]:
    """Extract text using pdfplumber."""
    text_parts = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return '\n'.join(text_parts), 'pdfplumber'


def _extract_with_pypdf(file_bytes: bytes) -> tuple[str, str]:
    """Extract text using pypdf."""
    text_parts = []

    pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return '\n'.join(text_parts), 'pypdf'


def extract_from_file(file_bytes: bytes, filename: str) -> Dict:
    """
    Main entry point for file extraction. Routes to appropriate handler.

    Currently supports: PDF
    TODO: Add support for DOC, DOCX, XLS, HTML

    Args:
        file_bytes: File content as bytes
        filename: Original filename

    Returns:
        Extraction result dict
    """
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        return extract_from_pdf(file_bytes, filename)
    else:
        return {
            'raw_text': '',
            'blocks': [],
            'method': 'unsupported',
            'success': False,
            'error': f'Unsupported file type. Currently only PDF is supported. Got: {filename}',
            'confidence': 'low'
        }
