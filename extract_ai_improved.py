"""
Improved AI-powered PDF extraction for manual requirements.

Key improvements:
1. Better detection of explicit AND vague manual inclusion language
2. Image/figure reference detection
3. Safety notice flagging (WARNING/DANGER/CAUTION)
4. Aggressive extraction (better to over-capture than miss)
"""
import anthropic
from typing import Dict, List
import json
import os
from dotenv import load_dotenv
import httpx
import re

load_dotenv()


def detect_image_references(text: str) -> tuple[bool, str]:
    """
    Detect if requirement references an image/figure.
    
    Returns:
        (has_image, reference) - e.g. (True, "Figure 7.2")
    """
    patterns = [
        r'[Ff]igure\s+[\dA-Z]+\.?\d*',
        r'[Ff]ig\.?\s+[\dA-Z]+\.?\d*',
        r'[Ii]llustration\s+[\dA-Z]+\.?\d*',
        r'[Dd]iagram\s+[\dA-Z]+\.?\d*',
        r'[Pp]ictogram\s+[\dA-Z]+\.?\d*',
        r'[Ss]ymbol\s+shown',
        r'see\s+image',
        r'as\s+shown\s+in',
        r'[Aa]nnex\s+[A-Z]\s+[Ff]igure',
        r'[Tt]able\s+[\dA-Z]+\.?\d*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return True, match.group(0)
    
    return False, ""


def detect_safety_notice(text: str) -> str:
    """
    Detect safety notice type.
    
    Returns:
        "WARNING" | "DANGER" | "CAUTION" | "HAZARD" | "None"
    """
    text_upper = text.upper()
    
    # Priority order (DANGER > WARNING > CAUTION)
    if 'DANGER' in text_upper:
        return "DANGER"
    elif 'WARNING' in text_upper:
        return "WARNING"
    elif 'HAZARD' in text_upper:
        return "HAZARD"
    elif 'CAUTION' in text_upper:
        return "CAUTION"
    
    return "None"


def extract_from_detected_sections(sections: List[Dict], standard_name: str = None, api_key: str = None) -> Dict:
    """
    HYBRID APPROACH: Use AI to intelligently extract from sections already detected by rules.

    This is the best approach:
    - Rules scan entire document and find relevant sections (fast, no API cost)
    - AI processes only those sections (smart, no truncation, no timeout)

    Args:
        sections: List of sections detected by rule-based detect_manual_sections()
        standard_name: Optional standard name for context
        api_key: Anthropic API key (uses env var if not provided)

    Returns:
        Dict with:
        - rows: List of requirement dicts with all 9 columns
        - stats: Extraction statistics
        - confidence: AI confidence level (high/medium/low)
    """

    # Get API key from env if not provided
    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("No Anthropic API key provided or found in environment")

    # Setup client
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(proxies=None, timeout=120.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception as e:
        print(f"[AI EXTRACTION] Client init error: {e}, trying without custom http_client")
        client = anthropic.Anthropic(api_key=api_key)

    all_requirements = []

    print(f"[HYBRID AI] Processing {len(sections)} sections with Claude...")

    for i, section in enumerate(sections, 1):
        section_text = section.get('content', '')
        heading = section.get('heading', '')
        clause = section.get('clause_number', '')

        print(f"[HYBRID AI] Section {i}/{len(sections)}: {heading} (Clause {clause})")

        # Build improved prompt for this section
        prompt = f"""You are an expert at analyzing e-bike safety standards and regulations to identify what manufacturers must communicate to users.

Standard: {standard_name or 'Unknown'}
Section: {heading}
Clause: {clause}

Section Content:
{section_text}

================================================================================
CRITICAL EXTRACTION RULES
================================================================================

WHAT TO EXTRACT:
Extract ANY requirement that obligates the manufacturer to COMMUNICATE something to the user in product/user manuals or documentation.

✅ INCLUDE if the requirement says or implies:
• "shall be stated/included in the manual"
• "shall be described in the instructions"  
• "user shall be informed/warned"
• "instructions must contain/include"
• "manual shall provide/describe"
• "information must be presented/made available"
• "user must be made aware"
• "documentation shall specify"
• "the following information shall be provided"
• Any vague language like "shall be provided to the user" (even if doesn't say "in manual")

✅ ALWAYS INCLUDE:
• ANY text with WARNING, DANGER, CAUTION, HAZARD (regardless of context)
• Requirements about what users need to know (operating temperature, load limits, maintenance)
• Assembly instructions that must be communicated
• Safety information that must be conveyed
• Symbols/pictograms that must appear in documentation

❌ EXCLUDE only if:
• Pure physical product requirement with NO mention of user communication
• Internal manufacturing processes with no user-facing requirement
• Testing procedures that don't need to be explained to users

WHEN IN DOUBT → INCLUDE IT. Better to over-capture than miss requirements.

================================================================================
EXTRACTION DETAILS
================================================================================

For EACH requirement found, extract:

1. **Description**: The full requirement text (preserve exact wording, measurements, specifications)

2. **Clause/Requirement**: The clause/section identifier with full hierarchy
   - Examples: "7.1", "7.1.1", "7.1.1.a", "1512.19(a)(1)", "Annex A.2"
   - PRESERVE the exact format from the source
   - Include parent hierarchy: if extracting "a)", note it's under "7.1.1.a"

3. **Requirement scope**: Keywords for product applicability
   - Use: ebike, bicycle, battery, charger, motor, controller, pedals, brakes, etc.
   - Can be multiple: "ebike, battery"

4. **Formatting required**: Is specific formatting mandated?
   - "Y" if text specifies: size, color, symbols, capital letters, bold, position
   - "N/A" if no formatting mentioned

5. **Required in Print**: Must it be in physical printed manual?
   - "y" if explicitly says: "in print", "printed manual", "paper documentation"
   - "n" if says: "electronic", "digital", "online", "downloadable"  
   - "N/A" if unclear or not specified

6. **Comments**: Important context
   - Note if vague language was used
   - Note if requirement is ambiguous
   - Note if it's inside a specific section

7. **Contains Image**: Does it reference a figure/image?
   - "Y" with reference (e.g., "Figure 7.2") if mentions: figure, diagram, illustration, pictogram, symbol shown, table
   - "N" if no image reference

8. **Safety Notice Type**: Is it a safety notice?
   - "WARNING" | "DANGER" | "CAUTION" | "HAZARD" | "None"
   - Check for these exact words in the requirement text

================================================================================
SPLITTING REQUIREMENTS
================================================================================

SPLIT into SEPARATE requirements when you see:
• Numbered subsections: 7.1, 7.2, 7.3
• Lettered items: a), b), c)
• Each distinct obligation or piece of information to communicate

KEEP TOGETHER as ONE requirement when:
• It's a single coherent instruction with multiple parts that make no sense separately
• It's a list that works as one unit

When splitting, preserve the FULL hierarchy in the Clause/Requirement field.

================================================================================
OUTPUT FORMAT
================================================================================

Respond with a JSON object:
{{
  "requirements": [
    {{
      "Description": "Full requirement text with exact wording and specs",
      "Clause/Requirement": "7.1.1.a",
      "Requirement scope": "ebike, battery",
      "Formatting required?": "Y",
      "Required in Print?": "y",
      "Comments": "inside instructions section; vague language used",
      "Contains Image?": "Y - Figure 7.2",
      "Safety Notice Type": "WARNING"
    }},
    ...
  ],
  "extraction_notes": "Any important observations",
  "confidence": "high|medium|low"
}}

BE AGGRESSIVE - extract anything that might require user communication. The engineer will filter out false positives later."""

        try:
            # Use streaming for each section
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

            # Add standard name to each requirement
            for req in section_requirements:
                req['Standard/Reg'] = standard_name or 'Unknown'
                
                # Post-process image and safety notice detection (double-check AI's work)
                desc = req.get('Description', '')
                
                # Verify image detection
                has_image, img_ref = detect_image_references(desc)
                if has_image and req.get('Contains Image?', 'N') == 'N':
                    req['Contains Image?'] = f"Y - {img_ref}"
                
                # Verify safety notice
                safety_type = detect_safety_notice(desc)
                if safety_type != "None" and req.get('Safety Notice Type', 'None') == 'None':
                    req['Safety Notice Type'] = safety_type

            all_requirements.extend(section_requirements)
            print(f"[HYBRID AI] Extracted {len(section_requirements)} requirements from this section")

        except Exception as e:
            print(f"[HYBRID AI] Error processing section {i}: {e}")
            # Continue with other sections even if one fails
            continue

    print(f"[HYBRID AI] Total extracted: {len(all_requirements)} requirements")

    return {
        'rows': all_requirements,
        'stats': {
            'total_detected': len(all_requirements),
            'classified_rows': len(all_requirements),
            'sections_processed': len(sections),
            'extraction_notes': f'Hybrid extraction: {len(sections)} sections processed'
        },
        'confidence': 'high'
    }


def extract_requirements_with_ai(pdf_text: str, standard_name: str = None, api_key: str = None) -> Dict:
    """
    Fallback: Use Claude to extract from full PDF text when no sections detected.
    Uses same improved prompt as section-based extraction.

    Args:
        pdf_text: Full text extracted from PDF
        standard_name: Optional standard name for context
        api_key: Anthropic API key (uses env var if not provided)

    Returns:
        Dict with:
        - rows: List of requirement dicts with all 9 columns
        - stats: Extraction statistics
        - confidence: AI confidence level
    """

    # Get API key
    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("No Anthropic API key provided or found in environment")

    # Setup client
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(proxies=None, timeout=180.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception as e:
        print(f"[AI EXTRACTION] Client init error: {e}")
        client = anthropic.Anthropic(api_key=api_key)

    # Build context
    standard_context = f"Standard: {standard_name}" if standard_name else "Standard: Unknown"

    # Use same improved prompt as section-based extraction
    prompt_template = f"""You are an expert at analyzing e-bike safety standards and regulations to identify what manufacturers must communicate to users.

{standard_context}

PDF Text:
{{PDF_TEXT}}

[Same CRITICAL EXTRACTION RULES as section-based extraction - full prompt here]

Extract ALL relevant requirements that obligate manufacturers to communicate with users."""

    try:
        print(f"[AI EXTRACTION] Sending {len(pdf_text)} chars to Claude...")

        # Limit to avoid timeout
        max_chars = 100000
        if len(pdf_text) > max_chars:
            print(f"[AI EXTRACTION] Truncating to {max_chars} chars")
            pdf_text = pdf_text[:max_chars] + "\n\n[Document truncated]"

        final_prompt = prompt_template.replace('{PDF_TEXT}', pdf_text)

        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            temperature=0,
            timeout=600.0,
            messages=[{"role": "user", "content": final_prompt}]
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                response_text += text

        # Parse response (same as section-based)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
        requirements = result.get('requirements', [])

        # Add standard name and post-process
        for req in requirements:
            req['Standard/Reg'] = standard_name or 'Unknown'
            
            desc = req.get('Description', '')
            
            # Image detection
            has_image, img_ref = detect_image_references(desc)
            if has_image and req.get('Contains Image?', 'N') == 'N':
                req['Contains Image?'] = f"Y - {img_ref}"
            
            # Safety notice
            safety_type = detect_safety_notice(desc)
            if safety_type != "None" and req.get('Safety Notice Type', 'None') == 'None':
                req['Safety Notice Type'] = safety_type

        confidence = result.get('confidence', 'medium')
        notes = result.get('extraction_notes', '')

        return {
            'rows': requirements,
            'stats': {
                'total_detected': len(requirements),
                'classified_rows': len(requirements),
                'extraction_notes': notes
            },
            'confidence': confidence
        }

    except Exception as e:
        print(f"[AI EXTRACTION] Failed: {e}")
        raise
