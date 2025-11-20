"""
GPT-4o-mini extraction for e-bike safety standards.

Minimal approach: Extract clause + text only, pipe through existing classify.py.
"""
import os
import json
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()


# Model configuration
MODEL_CONFIG = {
    "model": "gpt-4o-mini",
    "max_tokens": 16000,
    "temperature": 0,
    "timeout": 120.0,
    "cost_per_mtok_input": 0.15,   # $0.15 per million tokens
    "cost_per_mtok_output": 0.60   # $0.60 per million tokens
}


def _log_api_usage(input_tokens: int, output_tokens: int, latency_seconds: float):
    """Log API usage metrics for cost tracking."""
    config = MODEL_CONFIG
    cost_input = (input_tokens / 1_000_000) * config["cost_per_mtok_input"]
    cost_output = (output_tokens / 1_000_000) * config["cost_per_mtok_output"]
    total_cost = cost_input + cost_output

    print(f"[GPT API] Model: {config['model']} | "
          f"Tokens: {input_tokens:,}in + {output_tokens:,}out | "
          f"Cost: ${total_cost:.4f} | "
          f"Latency: {latency_seconds:.2f}s")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken."""
    try:
        # Use cl100k_base encoding (same as GPT-4 and GPT-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"[TIKTOKEN] Error: {e}, using rough estimate")
        return len(text) // 4  # Fallback: ~4 chars per token


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 6000,
    overlap_tokens: int = 500
) -> List[str]:
    """
    Chunk text by token count with overlap.

    Args:
        text: Full text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks to avoid boundary issues

    Returns:
        List of text chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move start forward, accounting for overlap
        start = end - overlap_tokens if end < len(tokens) else end

    return chunks


def chunk_text_smart(
    text: str,
    max_tokens: int = 6000,
    overlap_tokens: int = 500
) -> List[str]:
    """
    Smart chunking: Try to split on section boundaries, fallback to token-based.

    Uses simple heuristics to find natural boundaries (section headers).
    Does NOT filter - just suggests split points.

    Args:
        text: Full text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap for fallback chunking

    Returns:
        List of text chunks
    """
    # Try to find section boundaries
    # Common patterns: "5. Safety requirements", "7.1 General", etc.
    import re

    # Find potential section headers (number at start of line)
    section_pattern = r'\n(\d+(?:\.\d+)*)\s+[A-Z][^\n]{10,100}\n'
    matches = list(re.finditer(section_pattern, text))

    if len(matches) > 5:  # If we found reasonable section headers
        print(f"[CHUNKING] Found {len(matches)} section headers, using smart chunking")

        chunks = []
        current_chunk = ""
        current_tokens = 0

        # Split text by section boundaries
        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end]
            sections.append(section_text)

        # Group sections into chunks
        for section_text in sections:
            section_tokens = count_tokens(section_text)

            if section_tokens > max_tokens:
                # Section too big, flush current and split this section
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # Split large section by tokens
                chunks.extend(chunk_text_by_tokens(section_text, max_tokens, overlap_tokens))

            elif current_tokens + section_tokens > max_tokens:
                # Would exceed limit, flush current chunk
                chunks.append(current_chunk)
                current_chunk = section_text
                current_tokens = section_tokens

            else:
                # Accumulate
                current_chunk += section_text
                current_tokens += section_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    else:
        # No clear sections, use pure token-based chunking
        print(f"[CHUNKING] No clear sections found, using token-based chunking")
        return chunk_text_by_tokens(text, max_tokens, overlap_tokens)


def _extract_single_chunk(
    chunk_text: str,
    filename: str,
    extraction_type: str,
    client: OpenAI,
    chunk_index: int,
    total_chunks: int
) -> Tuple[int, List[Dict]]:
    """
    Extract requirements from a single chunk using GPT-4o-mini.

    Args:
        chunk_text: Text chunk to process
        filename: Source filename for context
        extraction_type: "manual" or "all"
        client: OpenAI client
        chunk_index: Index of this chunk
        total_chunks: Total number of chunks

    Returns:
        (chunk_index, list of requirements)
    """
    # Build extraction instructions based on type
    if extraction_type == "all":
        focus_instructions = """EXTRACT ALL NUMBERED CLAUSES INCLUDING:

1. **Clause headings** - for parent section mapping (e.g., "6.1 Vibration")
2. **Clause content with normative language** - shall/must/required
3. **Manual-related clauses** - instructions/markings/warnings (even without shall/must)
4. **Design, technical, and performance specifications**
5. **Test procedures, quality checks, and validation requirements**
6. **Manufacturing, production, and assembly requirements**
7. **Safety requirements, warnings, and notices**
8. **Installation, maintenance, and service requirements**
9. **Definitions, scope statements, and normative references**
10. **Tables, figures references, and appendices**

DO NOT FILTER. If it has a clause number, extract it.
Maintain hierarchical structure for parent section tracking."""
    else:
        focus_instructions = """Extract ONLY requirements that relate to user manuals, instruction documentation,
or information that must be communicated to users.

Include requirements about:
• What must be stated in manuals
• Information users need to know
• Warnings, cautions, safety notices
• Assembly/installation instructions
• Symbols and pictograms for documentation
• Marking and labeling requirements"""

    prompt = f"""You are extracting requirements from an e-bike safety standard.

DOCUMENT: {filename}
EXTRACTION MODE: {extraction_type}

{focus_instructions}

CRITICAL RULES (DO NOT DEVIATE):
1. Extract EVERY item with a clause/section number
2. Include numbered paragraphs even without "shall" (e.g., definitions, notes, examples)
3. Extract ISO modification markers: "Replacement:", "Addition:", "Amendment:"
4. Handle complex numbering: "5.1.101", "7.3.2.a.1", "Annex A.2.3.b"
5. Include lettered sub-items: (a), (b), (c), etc.
6. Include roman numerals: (i), (ii), (iii)
7. Keep FULL text - don't summarize or truncate
8. Extract table/figure titles if they have clause numbers
9. DO NOT skip items because they seem like headers or metadata
10. When in doubt, INCLUDE IT

OUTPUT FORMAT: Return ONLY a JSON array (no markdown, no explanations):
[
  {{"clause": "5.1.101", "text": "Full requirement text here..."}},
  {{"clause": "7.3.2.a", "text": "Another requirement..."}},
  ...
]

TEXT TO PROCESS:
{chunk_text}

Return JSON array only:"""

    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MODEL_CONFIG["max_tokens"],
            temperature=MODEL_CONFIG["temperature"],
            timeout=MODEL_CONFIG["timeout"]
        )

        latency = time.time() - start_time

        # Extract response text
        response_text = response.choices[0].message.content

        # Log API usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        _log_api_usage(input_tokens, output_tokens, latency)

        # Parse JSON
        requirements = []
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            requirements = json.loads(json_str)

            if not isinstance(requirements, list):
                print(f"[GPT] Chunk {chunk_index+1}/{total_chunks}: Expected array, got {type(requirements)}")
                requirements = []

            print(f"[GPT] Chunk {chunk_index+1}/{total_chunks}: Extracted {len(requirements)} requirements")

        except json.JSONDecodeError as e:
            print(f"[GPT] Chunk {chunk_index+1}/{total_chunks}: JSON parse error: {e}")
            print(f"[GPT] Response preview: {response_text[:200]}...")
            requirements = []

        return (chunk_index, requirements)

    except Exception as e:
        print(f"[GPT] Chunk {chunk_index+1}/{total_chunks}: Error: {e}")
        import traceback
        traceback.print_exc()
        return (chunk_index, [])


def extract_requirements_gpt(
    pdf_text: str,
    filename: str,
    extraction_type: str = "manual",
    api_key: str = None,
    max_workers: int = 5,
    progress_callback = None
) -> List[Dict]:
    """
    Extract requirements from PDF text using GPT-4o-mini.

    Returns minimal format: clause + text only.
    Classification happens downstream in classify.py.

    Args:
        pdf_text: Full text extracted from PDF
        filename: Source filename for context
        extraction_type: "manual" (user docs only) or "all" (everything)
        api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        max_workers: Number of parallel workers (default: 5)
        progress_callback: Optional callback for progress updates

    Returns:
        List of dicts: [{"clause": "5.1.101", "text": "..."}, ...]
    """
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY env var or pass api_key parameter.")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    mode_label = "ALL REQUIREMENTS" if extraction_type == "all" else "MANUAL REQUIREMENTS"
    print(f"[GPT EXTRACTION] Starting {mode_label} extraction for {filename}")
    print(f"[GPT EXTRACTION] Document length: {len(pdf_text):,} characters")

    # Chunk the text
    chunks = chunk_text_smart(pdf_text, max_tokens=6000, overlap_tokens=500)
    num_chunks = len(chunks)

    print(f"[GPT EXTRACTION] Split into {num_chunks} chunks")
    print(f"[GPT EXTRACTION] Using {max_workers} parallel workers")

    # Process chunks in parallel
    all_requirements = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                _extract_single_chunk,
                chunks[i],
                filename,
                extraction_type,
                client,
                i,
                num_chunks
            ): i
            for i in range(num_chunks)
        }

        completed_chunks = 0
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                idx, requirements = future.result(timeout=180)
                all_requirements.extend(requirements)

                # Progress logging
                completed_chunks += 1
                progress_pct = (completed_chunks / num_chunks) * 100

                print(f"[PROGRESS] {completed_chunks}/{num_chunks} chunks ({progress_pct:.1f}%) | "
                      f"{len(all_requirements)} requirements extracted")

                if progress_callback:
                    progress_callback(completed_chunks, num_chunks, len(all_requirements))

            except Exception as e:
                print(f"[GPT] Chunk {chunk_idx+1} failed: {e}")
                continue

    print(f"[GPT EXTRACTION] Total: {len(all_requirements)} requirements extracted (before deduplication)")

    # Smart deduplication: use text similarity for same clause IDs
    from difflib import SequenceMatcher

    unique_requirements = []
    duplicates_removed = 0

    for req in all_requirements:
        clause = req.get('clause', '').strip()
        text = req.get('text', '').strip()

        if not clause or not text:
            continue

        # Check if we already have this clause
        is_duplicate = False
        for existing in unique_requirements:
            if existing['clause'] == clause:
                # Same clause - check text similarity
                similarity = SequenceMatcher(None, existing['text'], text).ratio()
                if similarity > 0.90:  # 90% similar = duplicate
                    is_duplicate = True
                    duplicates_removed += 1
                    break

        if not is_duplicate:
            unique_requirements.append(req)

    if duplicates_removed > 0:
        print(f"[DEDUPLICATION] Removed {duplicates_removed} duplicates (90% text similarity threshold)")

    print(f"[GPT EXTRACTION] Final: {len(unique_requirements)} unique requirements")

    return unique_requirements


def extract_requirements_gpt_batched(
    pdf_text: str,
    filename: str,
    extraction_type: str = "manual",
    api_key: str = None,
    max_workers: int = 5,
    progress_callback = None
) -> Dict:
    """
    Extract requirements and return in format compatible with existing pipeline.

    Returns same structure as extract_ai.extract_from_detected_sections_batched().

    Args:
        pdf_text: Full text extracted from PDF
        filename: Source filename
        extraction_type: "manual" or "all"
        api_key: OpenAI API key
        max_workers: Parallel workers
        progress_callback: Progress callback

    Returns:
        Dict with 'rows', 'stats', 'confidence' keys (compatible with existing pipeline)
    """
    requirements = extract_requirements_gpt(
        pdf_text,
        filename,
        extraction_type,
        api_key,
        max_workers,
        progress_callback
    )

    return {
        'rows': requirements,
        'stats': {
            'total_detected': len(requirements),
            'classified_rows': len(requirements),
            'extraction_method': 'gpt-4o-mini'
        },
        'confidence': 'high'
    }
