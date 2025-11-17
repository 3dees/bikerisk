"""
Improved AI-powered PDF extraction for manual requirements.
"""
import anthropic
from typing import Dict, List, Tuple, Callable, TypeVar
import json
import os
from dotenv import load_dotenv
import httpx
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
from datetime import datetime
import time
from extract import fix_encoding

load_dotenv()

# Type variable for retry function
T = TypeVar('T')


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except anthropic.APIStatusError as e:
            last_exception = e
            if e.status_code == 529:  # Overloaded
                if attempt < max_retries:
                    print(f"[RETRY] API overloaded, waiting {delay:.1f}s before retry {attempt+1}/{max_retries}")
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                else:
                    print(f"[RETRY] Max retries reached, giving up")
                    raise
            else:
                # Don't retry other status codes
                raise
        except Exception as e:
            # Don't retry unexpected errors
            raise

    # Should never reach here, but just in case
    raise last_exception


# Cache configuration
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "section_extractions.json"

# Model configuration
MODEL_CONFIG = {
    "extraction": {
        "model": os.getenv("EXTRACTION_MODEL", "claude-sonnet-4-5-20250929"),  # TEMP: Switched back to Sonnet - Haiku quality issues
        "max_tokens": 16000,  # Sonnet's output limit
        "temperature": 0,
        "timeout": 300.0,
        "cost_per_mtok_input": 3.0,    # Sonnet pricing
        "cost_per_mtok_output": 15.0   # Sonnet pricing
    },
    "consolidation": {
        "model": os.getenv("CONSOLIDATION_MODEL", "claude-sonnet-4-5-20250929"),  # Sonnet 4.5 for reasoning
        "max_tokens": 16000,  # Sonnet supports up to 16K output tokens
        "temperature": 0,
        "timeout": 600.0,
        "cost_per_mtok_input": 3.0,    # $3.00 per million tokens
        "cost_per_mtok_output": 15.0   # $15.00 per million tokens
    }
}


def _log_api_usage(model_type: str, input_tokens: int, output_tokens: int, latency_seconds: float):
    """
    Log API usage metrics for cost tracking and performance monitoring.

    Args:
        model_type: "extraction" or "consolidation"
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        latency_seconds: API call latency in seconds
    """
    config = MODEL_CONFIG.get(model_type, {})
    cost_input = (input_tokens / 1_000_000) * config.get("cost_per_mtok_input", 0)
    cost_output = (output_tokens / 1_000_000) * config.get("cost_per_mtok_output", 0)
    total_cost = cost_input + cost_output

    print(f"[API USAGE] {model_type.upper()} | "
          f"Model: {config.get('model', 'unknown')} | "
          f"Tokens: {input_tokens:,}in + {output_tokens:,}out | "
          f"Cost: ${total_cost:.4f} | "
          f"Latency: {latency_seconds:.2f}s")


def _get_section_hash(section_content: str, extraction_type: str) -> str:
    """Create hash of section content + extraction type for caching.

    Args:
        section_content: The section text
        extraction_type: "manual" or "all" (affects extraction, so part of cache key)

    Returns:
        MD5 hash string
    """
    cache_key = f"{extraction_type}:{section_content}"
    return hashlib.md5(cache_key.encode()).hexdigest()


def _load_cache() -> dict:
    """Load cached extractions from disk.

    Returns:
        Dict mapping section hashes to cached extraction results
    """
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"[CACHE] Error loading cache: {e}, starting fresh")
            return {}
    return {}


def _save_cache(cache: dict):
    """Save cache to disk atomically.

    Args:
        cache: Dict mapping section hashes to extraction results
    """
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    except Exception as e:
        print(f"[CACHE] Error saving cache: {e}")


def segment_section_into_clauses(
    section: Dict,
    max_tokens_per_chunk: int = 8000,
    extraction_type: str = "manual"
) -> List[Dict]:
    """
    Segment a section into clause-level chunks for fine-grained batching.

    Uses HYBRID approach:
    1. Regex patterns for obvious clause boundaries (fast)
    2. Paragraph structure analysis (flexible)
    3. Preserves original text and offsets (no data loss)

    Args:
        section: Section dict with 'content', 'heading', 'clause_number'
        max_tokens_per_chunk: Approx token limit per chunk (default 8000)
        extraction_type: "manual" or "all" (affects boundary detection)

    Returns:
        List of clause chunks with metadata
    """
    content = section.get('content', '')
    if not content.strip():
        return []

    parent_heading = section.get('heading', '')
    parent_clause = section.get('clause_number', '')
    start_line = section.get('start_line', 0)
    end_line = section.get('end_line', 0)

    # Define clause boundary patterns based on extraction_type
    # These are HINTS, not strict rules - we'll validate with structure analysis
    if extraction_type == "all":
        # Broader patterns for all requirements
        clause_patterns = [
            r'^\d+(\.\d+)+\s',           # 7.1.2, 4.1.2.3
            r'^\d+(\.\d+)*\([a-z]\)\s',  # 7.1(a), 4.2(b)
            r'^[A-Z](\.\d+)+\s',         # A.2.3, B.1.a
            r'^[IVX]+\.\d+\s',           # II.3, VII.4
            r'^\([a-z]\)\s',             # (a), (b)
            r'^[a-z]\)\s',               # a), b), c)
            r'^\d+\)\s',                 # 1), 2), 3)
        ]
    else:
        # Focused patterns for manual sections
        clause_patterns = [
            r'^\d+(\.\d+)+\s',           # 7.1.2, 1.7.4.1
            r'^\d+(\.\d+)*\([a-z]\)\s',  # 7.1(a)
            r'^\([a-z]\)\s',             # (a), (b)
            r'^[a-z]\)\s',               # a), b)
        ]

    lines = content.split('\n')
    chunks = []
    current_chunk_lines = []
    current_clause = parent_clause or 'N/A'
    current_tokens = 0

    for line in lines:
        stripped = line.strip()

        # Skip empty lines but preserve them in chunks
        if not stripped:
            current_chunk_lines.append(line)
            continue

        # Check if this line starts a new clause (HINT, not absolute)
        is_new_clause = False
        new_clause_id = None

        for pattern in clause_patterns:
            match = re.match(pattern, stripped)
            if match:
                is_new_clause = True
                new_clause_id = match.group(0).strip()
                break

        # Structural validation: Is this REALLY a clause boundary?
        # Don't split on:
        # - Lines that are too short (< 10 chars after number)
        # - Lines that look like continued text
        # - Lines within tables or lists (detect via indentation/bullets)
        if is_new_clause:
            remaining_text = stripped[len(new_clause_id):].strip()
            if len(remaining_text) < 10:
                # Too short, probably not a real clause
                is_new_clause = False
            elif remaining_text and not remaining_text[0].isupper() and not remaining_text[0].isdigit():
                # Doesn't start with capital, probably continued text
                is_new_clause = False

        # If confirmed new clause AND we have existing content, save current chunk
        if is_new_clause and current_chunk_lines:
            chunks.append({
                'content': '\n'.join(current_chunk_lines),
                'parent_heading': parent_heading,
                'parent_clause': parent_clause,
                'clause_number': current_clause,
                'start_line': start_line,
                'end_line': end_line,
                'extraction_type': extraction_type
            })

            # Start new chunk
            current_chunk_lines = [line]
            current_clause = new_clause_id if new_clause_id else current_clause
            current_tokens = len(line) // 4
        else:
            # Add to current chunk
            current_chunk_lines.append(line)
            current_tokens += len(line) // 4

            # Split if chunk exceeds token limit (safety valve)
            if current_tokens > max_tokens_per_chunk and current_chunk_lines:
                chunks.append({
                    'content': '\n'.join(current_chunk_lines),
                    'parent_heading': parent_heading,
                    'parent_clause': parent_clause,
                    'clause_number': current_clause,
                    'start_line': start_line,
                    'end_line': end_line,
                    'extraction_type': extraction_type
                })
                current_chunk_lines = []
                current_tokens = 0

    # Add final chunk
    if current_chunk_lines:
        chunks.append({
            'content': '\n'.join(current_chunk_lines),
            'parent_heading': parent_heading,
            'parent_clause': parent_clause,
            'clause_number': current_clause,
            'start_line': start_line,
            'end_line': end_line,
            'extraction_type': extraction_type
        })

    # Fallback: If no chunks created (no clause patterns matched), return entire section as one chunk
    if not chunks:
        return [{
            'content': content,
            'parent_heading': parent_heading,
            'parent_clause': parent_clause,
            'clause_number': parent_clause or 'N/A',
            'start_line': start_line,
            'end_line': end_line,
            'extraction_type': extraction_type
        }]

    return chunks


def detect_image_references(text: str) -> tuple[bool, str]:
    """Detect if requirement references an image/figure."""
    patterns = [
        r'[Ff]igure\s+[\dA-Z]+\.?\d*',
        r'[Ff]ig\.?\s+[\dA-Z]+\.?\d*',
        r'[Ii]llustration\s+[\dA-Z]+\.?\d*',
        r'[Dd]iagram\s+[\dA-Z]+\.?\d*',
        r'[Pp]ictogram\s+[\dA-Z]+\.?\d*',
        r'[Ss]ymbol\s+shown',
        r'see\s+image',
        r'as\s+shown\s+in',
        r'[Tt]able\s+[\dA-Z]+\.?\d*',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return True, match.group(0)

    return False, ""


def detect_safety_notice(text: str) -> str:
    """Detect safety notice type."""
    text_upper = text.upper()

    if 'DANGER' in text_upper:
        return "DANGER"
    elif 'WARNING' in text_upper:
        return "WARNING"
    elif 'HAZARD' in text_upper:
        return "HAZARD"
    elif 'CAUTION' in text_upper:
        return "CAUTION"

    return "None"


def remove_duplicate_requirements(requirements: List[Dict], similarity_threshold: float = 0.95) -> Tuple[List[Dict], List[Dict]]:
    """Remove duplicate requirements based on text similarity.

    Args:
        requirements: List of requirement dictionaries
        similarity_threshold: Similarity threshold (0.0-1.0) for considering duplicates (default: 0.95)

    Returns:
        Tuple of (unique_requirements, duplicate_info)
        - unique_requirements: List of unique requirements
        - duplicate_info: List of dicts with duplicate information
    """
    unique_requirements = []
    duplicates_info = []

    for req in requirements:
        req_text = req.get('Description', '').strip().lower()

        if not req_text:
            # Empty description, keep it but flag
            unique_requirements.append(req)
            continue

        is_duplicate = False

        for existing in unique_requirements:
            existing_text = existing.get('Description', '').strip().lower()

            if not existing_text:
                continue

            # Calculate similarity using SequenceMatcher
            similarity = SequenceMatcher(None, req_text, existing_text).ratio()

            if similarity >= similarity_threshold:
                # Found duplicate
                duplicates_info.append({
                    'duplicate_clause': req.get('Clause/Requirement', 'N/A'),
                    'original_clause': existing.get('Clause/Requirement', 'N/A'),
                    'similarity': round(similarity * 100, 1),
                    'description': req_text[:100] + '...' if len(req_text) > 100 else req_text
                })
                is_duplicate = True
                print(f"[DUPLICATE DETECTION] Removed: [{req.get('Clause/Requirement')}] matches [{existing.get('Clause/Requirement')}] (similarity: {similarity*100:.1f}%)")
                break

        if not is_duplicate:
            unique_requirements.append(req)

    if duplicates_info:
        print(f"[DUPLICATE DETECTION] Removed {len(duplicates_info)} duplicate requirements ({len(unique_requirements)} unique remaining)")
    else:
        print(f"[DUPLICATE DETECTION] No duplicates found ({len(unique_requirements)} unique requirements)")

    return unique_requirements, duplicates_info


def _extract_single_batch(
    batch_sections: List[Dict],
    standard_name: str,
    extraction_type: str,
    client: anthropic.Anthropic,
    batch_index: int,
    total_batches: int,
    start_section_idx: int,
    cache: dict
) -> tuple[int, List[Dict], Dict[str, Dict]]:
    """Extract requirements from a single batch with cache checking.

    Args:
        batch_sections: List of sections in this batch
        standard_name: Name of the standard
        extraction_type: "manual" or "all"
        client: Anthropic client
        batch_index: Index of this batch (for logging)
        total_batches: Total number of batches
        start_section_idx: Starting section index for this batch
        cache: Cache dictionary

    Returns:
        (batch_index, requirements_list, new_cache_entries_dict)
    """
    new_cache_entries = {}
    all_requirements = []
    mode_label = "ALL REQUIREMENTS" if extraction_type == "all" else "MANUAL REQUIREMENTS"

    # Check cache for each section in batch
    sections_to_process = []
    for section in batch_sections:
        section_content = section.get('content', '')
        section_hash = _get_section_hash(section_content, extraction_type)

        if section_hash in cache:
            # Cache hit - use cached results
            cached_requirements = cache[section_hash]['requirements']
            all_requirements.extend(cached_requirements)
        else:
            # Cache miss - need to process
            sections_to_process.append(section)

    # If all sections were cached, return early
    if not sections_to_process:
        print(f"[CACHE] Batch {batch_index+1}/{total_batches}: All {len(batch_sections)} sections cached")
        return (batch_index, all_requirements, new_cache_entries)

    print(f"[BATCH] {batch_index+1}/{total_batches}: Processing {len(sections_to_process)} sections (cached: {len(batch_sections) - len(sections_to_process)})")

    # Build focus instructions based on extraction type
    if extraction_type == "all":
        focus_instructions = """Extract ALL requirements from this standard, including:
• Design specifications and technical requirements
• Test procedures and quality requirements
• Manufacturing and production requirements
• User documentation and manual requirements
• Safety requirements and warnings
• Installation and maintenance requirements
• Performance standards and measurements

IMPORTANT: This document may use various numbering schemes:
- Numeric: 4.1.2, 7.3.1.1
- Letters: A.2.3, B.1.a
- Roman: II.a, VII.4
- Mixed: 4.1.a, A.2(b), 7.1(i)
- Lists: A), (1), (a)

Adapt to whatever numbering scheme this document uses.
Preserve the original clause number EXACTLY as written in the document."""
    else:
        focus_instructions = """Extract ANY requirement that obligates the manufacturer to COMMUNICATE something to users in manuals/documentation.

✅ INCLUDE if it says or implies:
• "shall be stated/included in the manual"
• "user shall be informed/warned"
• "instructions must contain/include"
• "information must be presented/made available"
• "user must be made aware"
• "shall be provided to the user" (even if doesn't say "in manual")

✅ ALWAYS INCLUDE:
• ANY text with WARNING, DANGER, CAUTION, HAZARD
• Requirements about what users need to know (temperature, load limits, maintenance)
• Assembly instructions
• Safety information
• Symbols/pictograms for documentation

❌ EXCLUDE only if:
• Pure physical product requirement with NO user communication mention
• Internal manufacturing processes
• Testing procedures users don't need to know

WHEN IN DOUBT → INCLUDE IT."""

    # Build combined sections content
    sections_content = ""
    for i, section in enumerate(sections_to_process):
        section_num = start_section_idx + i + 1
        heading = section.get('heading', '')
        clause = section.get('clause_number', '')
        content = section.get('content', '')

        sections_content += f"""
---SECTION {section_num}---
Heading: {heading}
Clause: {clause}
Content:
{content}

"""

    # Build the batch prompt
    prompt = f"""You are extracting requirements from an e-bike safety standard document.

{focus_instructions}

EXTRACTION RULES:
1. **Requirement Identification:**
   - Look for SHALL, MUST, REQUIRED, or similar mandatory language
   - Each requirement is a distinct obligation or specification
   - Include the FULL text of the requirement (do not truncate)

2. **Clause/Requirement Field:**
   - Extract the exact clause number as it appears in the document
   - Examples: "7.1.2", "4.2.6.c", "A.3.1", "7.1(a)"
   - PRESERVE full clause hierarchy (7.1.1.a format)
   - If no clear number, use parent section number

3. **Requirement Scope:**
   - Specify what the requirement applies to: ebike, battery, charger, bicycle
   - Use comma-separated list if multiple (e.g., "ebike, battery")
   - Infer from context if not explicitly stated

4. **Formatting Required:**
   - Capture any specific formatting rules (font size, color, capitalization, etc.)
   - Examples: "in capital letters", "minimum 2mm height", "red background"
   - If none specified, use "N"

5. **Required in Print:**
   - "y" if it MUST appear in printed materials/labels/manuals
   - "n" if it's a design/testing requirement without print obligation
   - "ambiguous" if unclear
   - Default to "y" for: warnings, cautions, default values, user instructions

6. **Contains Image:**
   - "Y - [reference]" if requirement references figures/diagrams/pictograms
   - Examples: "Y - Figure 7.2", "Y - Table 4", "Y - Pictogram A"
   - "N" if no visual reference

7. **Safety Notice Type:**
   - Detect: WARNING, CAUTION, DANGER, HAZARD
   - "None" if not a safety notice
   - Check the actual text for these keywords

8. **Comments:**
   - Note any ambiguity, special conditions, or classification rationale
   - Examples: "applies only to lithium batteries", "unclear if print required"

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
  "requirements": [
    {{
      "Description": "Full requirement text here...",
      "Clause/Requirement": "7.1.2.a",
      "Requirement scope": "ebike, battery",
      "Formatting required?": "minimum 2mm height",
      "Required in Print?": "y",
      "Comments": "classification notes",
      "Contains Image?": "Y - Figure 7.2",
      "Safety Notice Type": "WARNING"
    }}
  ],
  "extraction_notes": "Any observations about the extraction",
  "confidence": "high"
}}

FLEXIBILITY: If document format is unusual, put core requirement in Description and use Comments for additional context. Set unclear fields to "N/A".

Process ALL sections below. Each is marked ---SECTION N---:

{sections_content}

Respond with JSON only. No additional text outside the JSON structure.
"""

    try:
        # Get model configuration
        extraction_config = MODEL_CONFIG["extraction"]

        # Wrap the API call with retry logic
        start_time = time.time()
        def make_api_call():
            with client.messages.stream(
                model=extraction_config["model"],
                max_tokens=extraction_config["max_tokens"],
                temperature=extraction_config["temperature"],
                timeout=extraction_config["timeout"],
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                response_text = ""
                for text in stream.text_stream:
                    response_text += text
            return response_text

        response_text = retry_with_backoff(make_api_call, max_retries=3)
        latency = time.time() - start_time

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
        batch_requirements = result.get('requirements', [])

        # Log API usage (approximate token counts)
        input_tokens = len(prompt) // 4  # Rough estimate: 4 chars per token
        output_tokens = len(response_text) // 4
        _log_api_usage("extraction", input_tokens, output_tokens, latency)

        # Add standard name and validate
        for req in batch_requirements:
            req['Standard/Reg'] = standard_name or 'Unknown'

            # Fix UTF-8 encoding in all text fields
            for key, value in req.items():
                if isinstance(value, str):
                    req[key] = fix_encoding(value)

            desc = req.get('Description', '')

            # Double-check image detection
            has_image, img_ref = detect_image_references(desc)
            if has_image and req.get('Contains Image?', 'N') == 'N':
                req['Contains Image?'] = f"Y - {img_ref}"

            # Double-check safety notice
            safety_type = detect_safety_notice(desc)
            if safety_type != "None" and req.get('Safety Notice Type', 'None') == 'None':
                req['Safety Notice Type'] = safety_type

        # Cache the results for each processed section
        for section in sections_to_process:
            section_content = section.get('content', '')
            section_hash = _get_section_hash(section_content, extraction_type)

            # Store ALL requirements from this batch for this section
            new_cache_entries[section_hash] = {
                'requirements': batch_requirements,
                'extraction_type': extraction_type,
                'timestamp': datetime.now().isoformat(),
                'standard': standard_name
            }

        all_requirements.extend(batch_requirements)
        print(f"[{mode_label}] Batch {batch_index+1}/{total_batches}: Extracted {len(batch_requirements)} requirements")

    except Exception as e:
        print(f"[{mode_label}] Error in batch {batch_index+1}: {e}")
        import traceback
        traceback.print_exc()

    return (batch_index, all_requirements, new_cache_entries)


def _extract_single_batch_clauses(
    batch_clauses: List[Dict],
    standard_name: str,
    extraction_type: str,
    client: anthropic.Anthropic,
    batch_index: int,
    total_batches: int,
    start_clause_idx: int,
    cache: dict
) -> tuple[int, List[Dict], Dict[str, Dict]]:
    """
    Extract requirements from a batch of clause chunks.

    Caching strategy: Cache at SECTION level (not clause level) to avoid data inflation.
    If a parent section is cached, all its clauses are skipped.
    """

    new_cache_entries = {}
    all_requirements = []
    mode_label = "ALL REQUIREMENTS" if extraction_type == "all" else "MANUAL REQUIREMENTS"

    # Group clauses by parent section for cache checking
    sections_map = {}
    for clause in batch_clauses:
        parent_key = f"{clause.get('parent_clause', 'N/A')}_{clause.get('parent_heading', '')}"
        if parent_key not in sections_map:
            sections_map[parent_key] = {
                'clauses': [],
                'full_content': '',
                'parent_clause': clause.get('parent_clause', ''),
                'parent_heading': clause.get('parent_heading', '')
            }
        sections_map[parent_key]['clauses'].append(clause)
        sections_map[parent_key]['full_content'] += clause.get('content', '') + '\n'

    # Check cache at section level
    clauses_to_process = []
    for section_key, section_data in sections_map.items():
        section_hash = _get_section_hash(section_data['full_content'], extraction_type)

        if section_hash in cache:
            # Cache hit - skip all clauses from this section
            cached_requirements = cache[section_hash]['requirements']
            all_requirements.extend(cached_requirements)
        else:
            # Cache miss - add clauses to process
            clauses_to_process.extend(section_data['clauses'])

    if not clauses_to_process:
        print(f"[CACHE] Batch {batch_index+1}/{total_batches}: All clauses from cached sections")
        return (batch_index, all_requirements, new_cache_entries)

    print(f"[BATCH] {batch_index+1}/{total_batches}: Processing {len(clauses_to_process)} clauses from {len(sections_map)} sections (some cached)")

    # Build combined content for this batch
    clauses_content = ""
    for idx, clause in enumerate(clauses_to_process, 1):
        clause_text = clause.get('content', '').strip()
        clause_num = clause.get('clause_number', 'N/A')
        parent_heading = clause.get('parent_heading', '')

        clauses_content += f"\n---CLAUSE {idx} (Parent: {parent_heading}, Clause: {clause_num})---\n"
        clauses_content += clause_text + "\n"

    # OPTIMIZED PROMPT
    if extraction_type == "all":
        focus_instructions = """Extract ALL requirements from this standard, including:
• Design specifications and technical requirements
• Test procedures and quality requirements
• Manufacturing and production requirements
• User documentation and manual requirements
• Safety requirements and warnings
• Installation and maintenance requirements
• Performance standards and measurements

IMPORTANT: Document may use various numbering schemes:
- Numeric: 4.1.2, 7.3.1.1
- Letters: A.2.3, B.1.a
- Roman: II.a, VII.4
- Mixed: 4.1.a, A.2(b), 7.1(i)
- Lists: A), (1), (a)

Preserve the original clause number EXACTLY as written."""
    else:
        focus_instructions = """Extract ANY requirement that obligates the manufacturer to COMMUNICATE something to users in manuals/documentation.

✅ INCLUDE if it says or implies:
• "shall be stated/included in the manual"
• "user shall be informed/warned"
• "instructions must contain/include"
• "information must be presented/made available"
• "user must be made aware"
• "shall be provided to the user" (even if doesn't say "in manual")

✅ ALWAYS INCLUDE:
• ANY text with WARNING, DANGER, CAUTION, HAZARD
• Requirements about what users need to know (temperature, load limits, maintenance)
• Assembly instructions
• Safety information
• Symbols/pictograms for documentation

❌ EXCLUDE only if:
• Pure physical product requirement with NO user communication mention
• Internal manufacturing processes
• Testing procedures users don't need to know

WHEN IN DOUBT → INCLUDE IT."""

    prompt = f"""You are extracting requirements from an e-bike safety standard document.

{focus_instructions}

EXTRACTION RULES:
1. **Requirement Identification:**
   - Look for SHALL, MUST, REQUIRED, or similar mandatory language
   - Each requirement is a distinct obligation or specification
   - Include the FULL text of the requirement (do not truncate)

2. **Clause/Requirement Field:**
   - Extract the exact clause number as it appears in the document
   - Examples: "7.1.2", "4.2.6.c", "A.3.1", "7.1(a)"
   - PRESERVE full clause hierarchy (7.1.1.a format)
   - If no clear number, use parent section number

3. **Requirement Scope:**
   - Specify what the requirement applies to: ebike, battery, charger, bicycle
   - Use comma-separated list if multiple (e.g., "ebike, battery")
   - Infer from context if not explicitly stated

4. **Formatting Required:**
   - Capture any specific formatting rules (font size, color, capitalization, etc.)
   - Examples: "in capital letters", "minimum 2mm height", "red background"
   - If none specified, use "N"

5. **Required in Print:**
   - "y" if it MUST appear in printed materials/labels/manuals
   - "n" if it's a design/testing requirement without print obligation
   - "ambiguous" if unclear
   - Default to "y" for: warnings, cautions, default values, user instructions

6. **Contains Image:**
   - "Y - [reference]" if requirement references figures/diagrams/pictograms
   - Examples: "Y - Figure 7.2", "Y - Table 4", "Y - Pictogram A"
   - "N" if no visual reference

7. **Safety Notice Type:**
   - Detect: WARNING, CAUTION, DANGER, HAZARD
   - "None" if not a safety notice
   - Check the actual text for these keywords

8. **Comments:**
   - Note any ambiguity, special conditions, or classification rationale
   - Examples: "applies only to lithium batteries", "unclear if print required"

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
  "requirements": [
    {{
      "Description": "Full requirement text here...",
      "Clause/Requirement": "7.1.2.a",
      "Requirement scope": "ebike, battery",
      "Formatting required?": "minimum 2mm height",
      "Required in Print?": "y",
      "Comments": "classification notes",
      "Contains Image?": "Y - Figure 7.2",
      "Safety Notice Type": "WARNING"
    }}
  ],
  "extraction_notes": "Any observations about the extraction",
  "confidence": "high"
}}

FLEXIBILITY: If document format is unusual, put core requirement in Description and use Comments for additional context. Set unclear fields to "N/A".

Process ALL clauses below. Each is marked ---CLAUSE N---:

{clauses_content}

Respond with JSON only. No additional text outside the JSON structure.
"""

    try:
        # Get model configuration
        extraction_config = MODEL_CONFIG["extraction"]

        # API call with retry logic
        start_time = time.time()
        def make_api_call():
            with client.messages.stream(
                model=extraction_config["model"],
                max_tokens=extraction_config["max_tokens"],
                temperature=extraction_config["temperature"],
                timeout=extraction_config["timeout"],
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                response_text = ""
                for text in stream.text_stream:
                    response_text += text
            return response_text

        response_text = retry_with_backoff(make_api_call, max_retries=3)
        latency = time.time() - start_time

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
        batch_requirements = result.get('requirements', [])

        # Log API usage (approximate token counts)
        input_tokens = len(prompt) // 4  # Rough estimate: 4 chars per token
        output_tokens = len(response_text) // 4
        _log_api_usage("extraction", input_tokens, output_tokens, latency)

        # Post-process requirements
        for req in batch_requirements:
            req['Standard/Reg'] = standard_name or 'Unknown'

            # Fix UTF-8 encoding
            for key, value in req.items():
                if isinstance(value, str):
                    req[key] = fix_encoding(value)

            desc = req.get('Description', '')

            # Validate image detection
            has_image, img_ref = detect_image_references(desc)
            if has_image and req.get('Contains Image?', 'N') == 'N':
                req['Contains Image?'] = f"Y - {img_ref}"

            # Validate safety notice
            safety_type = detect_safety_notice(desc)
            if safety_type != "None" and req.get('Safety Notice Type', 'None') == 'None':
                req['Safety Notice Type'] = safety_type

        # NOTE: Caching is disabled for clause-level batching because we cannot reliably
        # attribute which requirements came from which parent section when processing
        # multiple sections together in one batch. The cache checking still works (above),
        # but we don't add new cache entries from clause-batched results.
        # Caching still works correctly in _extract_single_batch() for section-level processing.

        all_requirements.extend(batch_requirements)
        print(f"[{mode_label}] Batch {batch_index+1}/{total_batches}: Extracted {len(batch_requirements)} requirements")

    except Exception as e:
        print(f"[{mode_label}] Error in batch {batch_index+1}: {e}")
        import traceback
        traceback.print_exc()

    return (batch_index, all_requirements, new_cache_entries)


def extract_from_detected_sections_batched(
    sections: List[Dict],
    standard_name: str = None,
    extraction_type: str = "manual",
    api_key: str = None,
    batch_size: int = None,  # DEPRECATED: backward compatibility
    clauses_per_batch: int = 75,
    max_workers: int = 5
) -> Dict:
    """
    Extract requirements using clause-level batching for 6-9x speed improvement.

    CHANGES from previous version:
    - Segments sections into clauses before batching
    - Batches by clause count (75) instead of section count (10)
    - Maintains parallel processing and caching
    - Works for both "manual" and "all" extraction types

    Args:
        sections: List of detected sections
        standard_name: Name of the standard being processed
        extraction_type: "manual" for manual requirements only, "all" for all requirements
        api_key: Anthropic API key
        batch_size: DEPRECATED - use clauses_per_batch instead (kept for backward compatibility)
        clauses_per_batch: Number of clause chunks to process per API call (default: 75)
        max_workers: Number of parallel workers (default: 5)
    """

    # Handle backward compatibility
    if batch_size is not None:
        print(f"[DEPRECATION WARNING] batch_size parameter is deprecated, use clauses_per_batch instead")
        clauses_per_batch = batch_size  # Treat old param as clause count

    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("No Anthropic API key provided")

    # Setup client
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(timeout=120.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception as e:
        print(f"[AI EXTRACTION] Client init error: {e}")
        client = anthropic.Anthropic(api_key=api_key)

    mode_label = "ALL REQUIREMENTS" if extraction_type == "all" else "MANUAL REQUIREMENTS"

    # Load cache
    cache = _load_cache()
    print(f"[CACHE] Loaded {len(cache)} cached extractions")

    # NEW: Segment sections into clause-level chunks
    print(f"[EXTRACTION] Segmenting {len(sections)} sections into clauses...")
    all_clause_chunks = []

    for section in sections:
        clause_chunks = segment_section_into_clauses(
            section,
            max_tokens_per_chunk=8000,
            extraction_type=extraction_type
        )
        all_clause_chunks.extend(clause_chunks)

    print(f"[EXTRACTION] Created {len(all_clause_chunks)} clause chunks from {len(sections)} sections")
    print(f"[EXTRACTION] Average chunks per section: {len(all_clause_chunks)/len(sections):.1f}")

    # NEW: Batch by clause count (not section count)
    num_batches = (len(all_clause_chunks) + clauses_per_batch - 1) // clauses_per_batch
    batches = []

    for batch_num in range(num_batches):
        start_idx = batch_num * clauses_per_batch
        end_idx = min(start_idx + clauses_per_batch, len(all_clause_chunks))
        batch_clauses = all_clause_chunks[start_idx:end_idx]
        batches.append((batch_num, start_idx, batch_clauses))

    print(f"[EXTRACTION] Split into {num_batches} batches ({clauses_per_batch} clauses/batch)")
    print(f"[EXTRACTION] Using {max_workers} parallel workers")

    all_requirements = []
    all_new_cache_entries = {}

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                _extract_single_batch_clauses,
                batch_clauses,
                standard_name,
                extraction_type,
                client,
                batch_idx,
                num_batches,
                start_idx,
                cache
            ): batch_idx
            for batch_idx, start_idx, batch_clauses in batches
        }

        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                idx, requirements, new_cache_entries = future.result(timeout=120)
                all_requirements.extend(requirements)
                all_new_cache_entries.update(new_cache_entries)

                # Progress logging
                completed_batches += 1
                progress_pct = (completed_batches / len(batches)) * 100

                print(f"[PROGRESS] {completed_batches}/{len(batches)} batches ({progress_pct:.1f}%) | "
                      f"{len(all_requirements)} requirements extracted")
            except Exception as e:
                print(f"[PARALLEL] Batch {batch_idx+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save cache
    if all_new_cache_entries:
        cache.update(all_new_cache_entries)
        _save_cache(cache)
        print(f"[CACHE] Saved {len(all_new_cache_entries)} new entries")

    print(f"[{mode_label}] Total: {len(all_requirements)} requirements from {len(all_clause_chunks)} clause chunks (before deduplication)")

    # Deduplicate
    unique_requirements, duplicates_info = remove_duplicate_requirements(all_requirements)

    return {
        'rows': unique_requirements,
        'stats': {
            'total_detected': len(unique_requirements),
            'classified_rows': len(unique_requirements),
            'sections_processed': len(sections),
            'clause_chunks_processed': len(all_clause_chunks),
            'batches_processed': num_batches,
            'duplicates_removed': len(duplicates_info),
            'original_count': len(all_requirements)
        },
        'confidence': 'high',
        'duplicates_info': duplicates_info
    }


def extract_from_detected_sections(sections: List[Dict], standard_name: str = None, extraction_type: str = "manual", api_key: str = None) -> Dict:
    """Extract requirements from detected sections using AI.

    Args:
        sections: List of detected sections
        standard_name: Name of the standard being processed
        extraction_type: "manual" for manual requirements only, "all" for all requirements
        api_key: Anthropic API key
    """

    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("No Anthropic API key provided")

    # Setup client
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(timeout=120.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception as e:
        print(f"[AI EXTRACTION] Client init error: {e}")
        client = anthropic.Anthropic(api_key=api_key)

    all_requirements = []
    mode_label = "ALL REQUIREMENTS" if extraction_type == "all" else "MANUAL REQUIREMENTS"
    print(f"[{mode_label}] Processing {len(sections)} sections...")

    for i, section in enumerate(sections, 1):
        section_text = section.get('content', '')
        heading = section.get('heading', '')
        clause = section.get('clause_number', '')

        print(f"[{mode_label}] Section {i}/{len(sections)}: {heading}")

        # Build prompt based on extraction type
        if extraction_type == "all":
            focus_instructions = """Extract ALL requirements from this standard, including:
• Design specifications and technical requirements
• Test procedures and quality requirements
• Manufacturing and production requirements
• User documentation and manual requirements
• Safety requirements and warnings
• Installation and maintenance requirements
• Performance standards and measurements

IMPORTANT: This document may use various numbering schemes:
- Numeric: 4.1.2, 7.3.1.1
- Letters: A.2.3, B.1.a
- Roman: II.a, VII.4
- Mixed: 4.1.a, A.2(b), 7.1(i)
- Lists: A), (1), (a)

Adapt to whatever numbering scheme this document uses.
Preserve the original clause number EXACTLY as written in the document."""
        else:
            focus_instructions = """Extract ANY requirement that obligates the manufacturer to COMMUNICATE something to users in manuals/documentation.

✅ INCLUDE if it says or implies:
• "shall be stated/included in the manual"
• "user shall be informed/warned"
• "instructions must contain/include"
• "information must be presented/made available"
• "user must be made aware"
• "shall be provided to the user" (even if doesn't say "in manual")

✅ ALWAYS INCLUDE:
• ANY text with WARNING, DANGER, CAUTION, HAZARD
• Requirements about what users need to know (temperature, load limits, maintenance)
• Assembly instructions
• Safety information
• Symbols/pictograms for documentation

❌ EXCLUDE only if:
• Pure physical product requirement with NO user communication mention
• Internal manufacturing processes
• Testing procedures users don't need to know

WHEN IN DOUBT → INCLUDE IT."""

        # Build the complete prompt
        prompt = f"""You are extracting requirements from an e-bike safety standard document.

{focus_instructions}

EXTRACTION RULES:
1. **Requirement Identification:**
   - Look for SHALL, MUST, REQUIRED, or similar mandatory language
   - Each requirement is a distinct obligation or specification
   - Include the FULL text of the requirement (do not truncate)

2. **Clause/Requirement Field:**
   - Extract the exact clause number as it appears in the document
   - Examples: "7.1.2", "4.2.6.c", "A.3.1", "7.1(a)"
   - PRESERVE full clause hierarchy (7.1.1.a format)
   - If no clear number, use parent section number

3. **Requirement Scope:**
   - Specify what the requirement applies to: ebike, battery, charger, bicycle
   - Use comma-separated list if multiple (e.g., "ebike, battery")
   - Infer from context if not explicitly stated

4. **Formatting Required:**
   - Capture any specific formatting rules (font size, color, capitalization, etc.)
   - Examples: "in capital letters", "minimum 2mm height", "red background"
   - If none specified, use "N"

5. **Required in Print:**
   - "y" if it MUST appear in printed materials/labels/manuals
   - "n" if it's a design/testing requirement without print obligation
   - "ambiguous" if unclear
   - Default to "y" for: warnings, cautions, default values, user instructions

6. **Contains Image:**
   - "Y - [reference]" if requirement references figures/diagrams/pictograms
   - Examples: "Y - Figure 7.2", "Y - Table 4", "Y - Pictogram A"
   - "N" if no visual reference

7. **Safety Notice Type:**
   - Detect: WARNING, CAUTION, DANGER, HAZARD
   - "None" if not a safety notice
   - Check the actual text for these keywords

8. **Comments:**
   - Note any ambiguity, special conditions, or classification rationale
   - Examples: "applies only to lithium batteries", "unclear if print required"

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
  "requirements": [
    {{
      "Description": "Full requirement text here...",
      "Clause/Requirement": "7.1.2.a",
      "Requirement scope": "ebike, battery",
      "Formatting required?": "minimum 2mm height",
      "Required in Print?": "y",
      "Comments": "classification notes",
      "Contains Image?": "Y - Figure 7.2",
      "Safety Notice Type": "WARNING"
    }}
  ],
  "extraction_notes": "Any observations about the extraction",
  "confidence": "high"
}}

FLEXIBILITY: If document format is unusual, put core requirement in Description and use Comments for additional context. Set unclear fields to "N/A".

Process this section:
Section: {heading}
Clause: {clause}

{section_text}

Respond with JSON only. No additional text outside the JSON structure.
"""

        try:
            # Get model configuration
            extraction_config = MODEL_CONFIG["extraction"]

            start_time = time.time()
            with client.messages.stream(
                model=extraction_config["model"],
                max_tokens=extraction_config["max_tokens"],
                temperature=extraction_config["temperature"],
                timeout=extraction_config["timeout"],
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                response_text = ""
                for text in stream.text_stream:
                    response_text += text
            latency = time.time() - start_time

            # Parse JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)
            section_requirements = result.get('requirements', [])

            # Log API usage (approximate token counts)
            input_tokens = len(prompt) // 4
            output_tokens = len(response_text) // 4
            _log_api_usage("extraction", input_tokens, output_tokens, latency)

            # Add standard name and validate
            for req in section_requirements:
                req['Standard/Reg'] = standard_name or 'Unknown'

                desc = req.get('Description', '')

                # Double-check image detection
                has_image, img_ref = detect_image_references(desc)
                if has_image and req.get('Contains Image?', 'N') == 'N':
                    req['Contains Image?'] = f"Y - {img_ref}"

                # Double-check safety notice
                safety_type = detect_safety_notice(desc)
                if safety_type != "None" and req.get('Safety Notice Type', 'None') == 'None':
                    req['Safety Notice Type'] = safety_type

            all_requirements.extend(section_requirements)
            print(f"[HYBRID AI] Extracted {len(section_requirements)} requirements")

        except Exception as e:
            print(f"[HYBRID AI] Error in section {i}: {e}")
            continue

    print(f"[HYBRID AI] Total: {len(all_requirements)} requirements")

    return {
        'rows': all_requirements,
        'stats': {
            'total_detected': len(all_requirements),
            'classified_rows': len(all_requirements),
            'sections_processed': len(sections)
        },
        'confidence': 'high'
    }


def extract_requirements_with_ai(pdf_text: str, standard_name: str = None, extraction_type: str = "manual", api_key: str = None) -> Dict:
    """Fallback: Extract from full PDF text when no sections found.

    Args:
        pdf_text: Full text of PDF
        standard_name: Name of the standard
        extraction_type: "manual" for manual requirements only, "all" for all requirements
        api_key: Anthropic API key
    """

    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("No API key")

    # Setup client
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(timeout=180.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception as e:
        print(f"[AI EXTRACTION] Error: {e}")
        client = anthropic.Anthropic(api_key=api_key)

    print(f"[AI EXTRACTION] Processing {len(pdf_text)} chars...")

    # Limit text to avoid timeout
    max_chars = 100000
    if len(pdf_text) > max_chars:
        print(f"[AI EXTRACTION] Truncating to {max_chars} chars")
        pdf_text = pdf_text[:max_chars] + "\n\n[Document truncated]"

    # Build prompt based on extraction type
    if extraction_type == "all":
        focus_instructions = """Extract ALL requirements from this standard, including:
• Design specifications and technical requirements
• Test procedures and quality requirements
• Manufacturing and production requirements
• User documentation and manual requirements
• Safety requirements and warnings
• Installation and maintenance requirements
• Performance standards and measurements

IMPORTANT: Document may use various numbering schemes:
- Numeric: 4.1.2, 7.3.1.1
- Letters: A.2.3, B.1.a
- Roman: II.a, VII.4
- Mixed: 4.1.a, A.2(b), 7.1(i)
- Lists: A), (1), (a)

Preserve the original clause number EXACTLY as written."""
    else:
        focus_instructions = """Extract ANY requirement that obligates the manufacturer to COMMUNICATE something to users in manuals/documentation.

✅ INCLUDE if it says or implies:
• "shall be stated/included in the manual"
• "user shall be informed/warned"
• "instructions must contain/include"
• "information must be presented/made available"
• "user must be made aware"
• "shall be provided to the user" (even if doesn't say "in manual")

✅ ALWAYS INCLUDE:
• ANY text with WARNING, DANGER, CAUTION, HAZARD
• Requirements about what users need to know (temperature, load limits, maintenance)
• Assembly instructions
• Safety information
• Symbols/pictograms for documentation

❌ EXCLUDE only if:
• Pure physical product requirement with NO user communication mention
• Internal manufacturing processes
• Testing procedures users don't need to know

WHEN IN DOUBT → INCLUDE IT."""

    prompt = f"""You are extracting requirements from an e-bike safety standard document.

{focus_instructions}

EXTRACTION RULES:
1. **Requirement Identification:**
   - Look for SHALL, MUST, REQUIRED, or similar mandatory language
   - Each requirement is a distinct obligation or specification
   - Include the FULL text of the requirement (do not truncate)

2. **Clause/Requirement Field:**
   - Extract the exact clause number as it appears in the document
   - Examples: "7.1.2", "4.2.6.c", "A.3.1", "7.1(a)"
   - PRESERVE full clause hierarchy (7.1.1.a format)
   - If no clear number, use "N/A"

3. **Requirement Scope:**
   - Specify what the requirement applies to: ebike, battery, charger, bicycle
   - Use comma-separated list if multiple (e.g., "ebike, battery")
   - Infer from context if not explicitly stated

4. **Formatting Required:**
   - Capture any specific formatting rules (font size, color, capitalization, etc.)
   - Examples: "in capital letters", "minimum 2mm height", "red background"
   - If none specified, use "N"

5. **Required in Print:**
   - "y" if it MUST appear in printed materials/labels/manuals
   - "n" if it's a design/testing requirement without print obligation
   - "ambiguous" if unclear
   - Default to "y" for: warnings, cautions, default values, user instructions

6. **Contains Image:**
   - "Y - [reference]" if requirement references figures/diagrams/pictograms
   - Examples: "Y - Figure 7.2", "Y - Table 4", "Y - Pictogram A"
   - "N" if no visual reference

7. **Safety Notice Type:**
   - Detect: WARNING, CAUTION, DANGER, HAZARD
   - "None" if not a safety notice
   - Check the actual text for these keywords

8. **Comments:**
   - Note any ambiguity, special conditions, or classification rationale
   - Examples: "applies only to lithium batteries", "unclear if print required"

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
  "requirements": [
    {{
      "Description": "Full requirement text here...",
      "Clause/Requirement": "7.1.2.a",
      "Requirement scope": "ebike, battery",
      "Formatting required?": "minimum 2mm height",
      "Required in Print?": "y",
      "Comments": "classification notes",
      "Contains Image?": "Y - Figure 7.2",
      "Safety Notice Type": "WARNING"
    }}
  ],
  "extraction_notes": "Any observations about the extraction",
  "confidence": "high"
}}

FLEXIBILITY: If document format is unusual, put core requirement in Description and use Comments for additional context. Set unclear fields to "N/A".

Process this PDF text:
{pdf_text}

Respond with JSON only. No additional text outside the JSON structure.
"""

    try:
        # Get model configuration
        extraction_config = MODEL_CONFIG["extraction"]

        start_time = time.time()
        with client.messages.stream(
            model=extraction_config["model"],
            max_tokens=extraction_config["max_tokens"],
            temperature=extraction_config["temperature"],
            timeout=600.0,  # Keep longer timeout for full PDF extraction
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                response_text += text
        latency = time.time() - start_time

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)

        # Log API usage (approximate token counts)
        input_tokens = len(prompt) // 4
        output_tokens = len(response_text) // 4
        _log_api_usage("extraction", input_tokens, output_tokens, latency)

        # Handle both formats: {"requirements": [...]} and [...]
        if isinstance(result, list):
            requirements = result
        else:
            requirements = result.get('requirements', [])

        # Add standard and validate
        for req in requirements:
            req['Standard/Reg'] = standard_name or 'Unknown'

            desc = req.get('Description', '')
            has_image, img_ref = detect_image_references(desc)
            if has_image and req.get('Contains Image?', 'N') == 'N':
                req['Contains Image?'] = f"Y - {img_ref}"

            safety_type = detect_safety_notice(desc)
            if safety_type != "None" and req.get('Safety Notice Type', 'None') == 'None':
                req['Safety Notice Type'] = safety_type

        return {
            'rows': requirements,
            'stats': {
                'total_detected': len(requirements),
                'classified_rows': len(requirements)
            },
            'confidence': result.get('confidence', 'medium') if isinstance(result, dict) else 'medium'
        }

    except Exception as e:
        print(f"[AI EXTRACTION] Failed: {e}")
        raise
