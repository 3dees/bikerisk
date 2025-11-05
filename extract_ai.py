"""
Improved AI-powered PDF extraction for manual requirements.
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


def extract_from_detected_sections(sections: List[Dict], standard_name: str = None, api_key: str = None) -> Dict:
    """Extract requirements from detected sections using AI."""

    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("No Anthropic API key provided")

    # Setup client
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(proxies=None, timeout=120.0)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception as e:
        print(f"[AI EXTRACTION] Client init error: {e}")
        client = anthropic.Anthropic(api_key=api_key)

    all_requirements = []
    print(f"[HYBRID AI] Processing {len(sections)} sections...")

    for i, section in enumerate(sections, 1):
        section_text = section.get('content', '')
        heading = section.get('heading', '')
        clause = section.get('clause_number', '')

        print(f"[HYBRID AI] Section {i}/{len(sections)}: {heading}")

        # THE IMPROVED PROMPT
        prompt = f"""You are an expert at analyzing e-bike safety standards to identify what manufacturers must communicate to users.

Standard: {standard_name or 'Unknown'}
Section: {heading}
Clause: {clause}

Section Content:
{section_text}

EXTRACTION RULES:

Extract ANY requirement that obligates the manufacturer to COMMUNICATE something to users in manuals/documentation.

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

WHEN IN DOUBT → INCLUDE IT.

For EACH requirement, extract:
1. Description: Full requirement text
2. Clause/Requirement: Clause ID with full hierarchy (e.g., "7.1.1.a")
3. Requirement scope: Keywords (ebike, battery, charger, etc.)
4. Formatting required?: "Y" if specific formatting specified, else "N/A"
5. Required in Print?: "y" if print required, "n" if digital OK, "N/A" if unclear
6. Comments: Note if vague language, ambiguous, etc.
7. Contains Image?: "Y - [reference]" if mentions figure/diagram, else "N"
8. Safety Notice Type: "WARNING" | "DANGER" | "CAUTION" | "HAZARD" | "None"

SPLIT numbered/lettered subsections into SEPARATE requirements.
PRESERVE full clause hierarchy.

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


def extract_requirements_with_ai(pdf_text: str, standard_name: str = None, api_key: str = None) -> Dict:
    """Fallback: Extract from full PDF text when no sections found."""

    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("No API key")

    # Setup client
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        http_client = httpx.Client(proxies=None, timeout=180.0)
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

    # Use same prompt as section-based extraction
    prompt = f"""Extract manual requirements from this e-bike standard.

Standard: {standard_name or 'Unknown'}

PDF Text:
{pdf_text}

[Use same rules as section-based extraction]

Extract requirements that obligate manufacturers to communicate with users.
BE AGGRESSIVE - include anything that might be user communication.
Extract 9 fields per requirement including "Contains Image?" and "Safety Notice Type".

Respond with JSON."""

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
            'confidence': result.get('confidence', 'medium')
        }

    except Exception as e:
        print(f"[AI EXTRACTION] Failed: {e}")
        raise
