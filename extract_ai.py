"""
Improved AI-powered PDF extraction for manual requirements.
"""
import anthropic
from typing import Dict, List, Tuple
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

load_dotenv()


# Cache configuration
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "section_extractions.json"


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
    prompt = f"""You are an expert at analyzing e-bike safety standards.

Standard: {standard_name or 'Unknown'}

EXTRACTION RULES:

{focus_instructions}

For EACH requirement, extract:
1. Description: Full requirement text
2. Clause/Requirement: Clause ID with full hierarchy (e.g., "7.1.1.a")
3. Requirement scope: Keywords (ebike, battery, charger, etc.)
4. Formatting required?: "Y" if specific formatting specified, else "N/A"
5. Required in Print?: "y" if print required, "n" if digital OK, "N/A" if unclear
6. Comments: Note if vague language, ambiguous, etc. USE THIS FIELD for any additional context that doesn't fit elsewhere
7. Contains Image?: "Y - [reference]" if mentions figure/diagram, else "N"
8. Safety Notice Type: "WARNING" | "DANGER" | "CAUTION" | "HAZARD" | "None"

SPLIT numbered/lettered subsections into SEPARATE requirements.
PRESERVE full clause hierarchy.

FLEXIBILITY: If document format is unusual, put core requirement in Description and use Comments for additional context. Set unclear fields to "N/A".

Process ALL sections below. Each is marked ---SECTION N---:

{sections_content}

Respond with JSON:
{{
  "requirements": [
    {{
      "Description": "Full text",
      "Clause/Requirement": "7.1.1.a",
      "Requirement scope": "ebike, battery",
      "Formatting required?": "Y",
      "Required in Print?": "y",
      "Comments": "vague language used",
      "Contains Image?": "Y - Figure 7.2",
      "Safety Notice Type": "WARNING"
    }}
  ],
  "extraction_notes": "Observations",
  "confidence": "high|medium|low"
}}"""

    try:
        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            temperature=0,
            timeout=300.0,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                response_text += text

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
        batch_requirements = result.get('requirements', [])

        # Add standard name and validate
        for req in batch_requirements:
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


def extract_from_detected_sections_batched(sections: List[Dict], standard_name: str = None, extraction_type: str = "manual", api_key: str = None, batch_size: int = 10) -> Dict:
    """Extract requirements from detected sections using AI with batch processing.

    This function processes multiple sections per API call for 10x efficiency improvement.

    Args:
        sections: List of detected sections
        standard_name: Name of the standard being processed
        extraction_type: "manual" for manual requirements only, "all" for all requirements
        api_key: Anthropic API key
        batch_size: Number of sections to process per API call (default: 10)
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
    all_new_cache_entries = {}
    mode_label = "ALL REQUIREMENTS" if extraction_type == "all" else "MANUAL REQUIREMENTS"

    # Load cache
    cache = _load_cache()
    print(f"[CACHE] Loaded {len(cache)} cached extractions")

    # Split sections into batches
    num_batches = (len(sections) + batch_size - 1) // batch_size
    batches = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(sections))
        batch_sections = sections[start_idx:end_idx]
        batches.append((batch_num, start_idx, batch_sections))

    print(f"[PARALLEL] Processing {len(sections)} sections in {num_batches} batches (batch_size={batch_size}) with 5 parallel workers...")

    # Process batches in parallel
    max_workers = 5
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(
                _extract_single_batch,
                batch_sections,
                standard_name,
                extraction_type,
                client,
                batch_idx,
                num_batches,
                start_idx,
                cache
            ): batch_idx
            for batch_idx, start_idx, batch_sections in batches
        }

        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                idx, requirements, new_cache_entries = future.result(timeout=120)
                all_requirements.extend(requirements)
                all_new_cache_entries.update(new_cache_entries)
            except Exception as e:
                print(f"[PARALLEL] Batch {batch_idx+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save new cache entries
    if all_new_cache_entries:
        cache.update(all_new_cache_entries)
        _save_cache(cache)
        print(f"[CACHE] Saved {len(all_new_cache_entries)} new entries")

    print(f"[{mode_label}] Total: {len(all_requirements)} requirements from {len(sections)} sections (before deduplication)")

    # Remove duplicates
    unique_requirements, duplicates_info = remove_duplicate_requirements(all_requirements)

    return {
        'rows': unique_requirements,
        'stats': {
            'total_detected': len(unique_requirements),
            'classified_rows': len(unique_requirements),
            'sections_processed': len(sections),
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
        prompt = f"""You are an expert at analyzing e-bike safety standards.

Standard: {standard_name or 'Unknown'}
Section: {heading}
Clause: {clause}

Section Content:
{section_text}

EXTRACTION RULES:

{focus_instructions}

For EACH requirement, extract:
1. Description: Full requirement text
2. Clause/Requirement: Clause ID with full hierarchy (e.g., "7.1.1.a")
3. Requirement scope: Keywords (ebike, battery, charger, etc.)
4. Formatting required?: "Y" if specific formatting specified, else "N/A"
5. Required in Print?: "y" if print required, "n" if digital OK, "N/A" if unclear
6. Comments: Note if vague language, ambiguous, etc. USE THIS FIELD for any additional context that doesn't fit elsewhere
7. Contains Image?: "Y - [reference]" if mentions figure/diagram, else "N"
8. Safety Notice Type: "WARNING" | "DANGER" | "CAUTION" | "HAZARD" | "None"

SPLIT numbered/lettered subsections into SEPARATE requirements.
PRESERVE full clause hierarchy.

FLEXIBILITY: If document format is unusual, put core requirement in Description and use Comments for additional context. Set unclear fields to "N/A".

Respond with JSON:
{{
  "requirements": [
    {{
      "Description": "Full text",
      "Clause/Requirement": "7.1.1.a",
      "Requirement scope": "ebike, battery",
      "Formatting required?": "Y",
      "Required in Print?": "y",
      "Comments": "vague language used",
      "Contains Image?": "Y - Figure 7.2",
      "Safety Notice Type": "WARNING"
    }}
  ],
  "extraction_notes": "Observations",
  "confidence": "high|medium|low"
}}"""

        try:
            with client.messages.stream(
                model="claude-sonnet-4-5-20250929",
                max_tokens=8000,
                temperature=0,
                timeout=300.0,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                response_text = ""
                for text in stream.text_stream:
                    response_text += text

            # Parse JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)
            section_requirements = result.get('requirements', [])

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
        focus = """Extract ALL requirements from this standard, including:
• Design specifications and technical requirements
• Test procedures and quality requirements
• Manufacturing and production requirements
• User documentation and manual requirements
• Safety requirements and warnings
• Installation and maintenance requirements

The document may use various numbering schemes (4.1.2, A.2, II.a, etc).
Preserve the exact clause number as written.
BE AGGRESSIVE - include anything that looks like a requirement."""
    else:
        focus = """Extract requirements that obligate manufacturers to communicate with users in manuals/documentation.
BE AGGRESSIVE - include anything that might be user communication."""

    prompt = f"""Extract requirements from this e-bike standard.

Standard: {standard_name or 'Unknown'}

PDF Text:
{pdf_text}

{focus}

For each requirement, try to extract these fields:
- Description: Main requirement text
- Clause/Requirement: Section/clause number if available
- Requirement scope: Keywords like "ebike", "battery", "charger"
- Formatting required?: "Y" if specific format mentioned, else "N/A"
- Required in Print?: "y" if must be printed, "n" if digital OK, "N/A" if unclear
- Comments: Any notes about vague language, ambiguity, or context
- Contains Image?: "Y - [reference]" if mentions figure/diagram, else "N"
- Safety Notice Type: "WARNING", "DANGER", "CAUTION", "HAZARD", or "None"

IMPORTANT: If the document format doesn't fit these fields well, you can:
1. Put the core requirement text in "Description"
2. Use "Comments" field to add any additional context or information that doesn't fit elsewhere
3. Set unclear fields to "N/A" rather than leaving empty
4. Adapt flexibly - these fields are guidelines, not strict rules

Respond with JSON in this format:
{{
  "requirements": [
    {{
      "Description": "...",
      "Clause/Requirement": "...",
      "Requirement scope": "...",
      "Formatting required?": "...",
      "Required in Print?": "...",
      "Comments": "...",
      "Contains Image?": "...",
      "Safety Notice Type": "..."
    }}
  ]
}}"""

    try:
        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            temperature=0,
            timeout=600.0,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                response_text += text

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)

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
