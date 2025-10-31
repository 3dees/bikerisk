"""
AI-powered PDF extraction using Claude.
Much simpler and more robust than rule-based extraction.
"""
import anthropic
from typing import Dict, List
import json
import os
from dotenv import load_dotenv
import httpx

load_dotenv()


def extract_requirements_with_ai(pdf_text: str, standard_name: str = None, api_key: str = None) -> Dict:
    """
    Use Claude to intelligently extract instruction/manual requirements from PDF text.

    Args:
        pdf_text: Raw text extracted from PDF
        standard_name: Optional standard name for context
        api_key: Anthropic API key (uses env var if not provided)

    Returns:
        Dict with:
        - rows: List of requirement dicts with all 7 columns
        - stats: Extraction statistics
        - confidence: AI confidence level (high/medium/low)
    """

    # Get API key from env if not provided
    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("No Anthropic API key provided or found in environment")

    # Create httpx client that respects proxy settings
    # Anthropic API should not go through proxy, so we add it to NO_PROXY
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

    try:
        # Initialize client with explicit http_client to avoid proxy issues
        http_client = httpx.Client(
            proxies=None,  # Explicitly disable proxies for Anthropic API
            timeout=120.0
        )
        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=http_client
        )
    except Exception as e:
        print(f"[AI EXTRACTION] Client init error: {e}, trying without custom http_client")
        # Fallback: try simple initialization
        client = anthropic.Anthropic(api_key=api_key)

    standard_context = f"Standard: {standard_name}\n" if standard_name else ""

    prompt = f"""You are an expert at analyzing e-bike safety standards and regulations.

{standard_context}
Your task: Extract ALL requirements related to user manuals, instruction manuals, warnings, documentation, and information that must be provided to users.

PDF Text:
{pdf_text[:100000]}

IMPORTANT GUIDELINES:
1. Look for requirements about what must be INCLUDED in user manuals/instructions
2. Include requirements about:
   - Manual content (what information must be provided)
   - Warnings and cautions
   - Safety instructions
   - Marking requirements (if they appear on the manual)
   - Documentation requirements
   - Information for use

3. EXCLUDE requirements about:
   - Manual operation (vs automatic operation)
   - Warning devices (vs warning text)
   - Physical product requirements unrelated to documentation

4. For EACH requirement found, extract:
   - **Description**: The full requirement text
   - **Clause**: The clause/section number (e.g., "7.1", "19.a.1", "a)")
   - **Requirement scope**: Keywords like "ebike, battery, charger" (what product types this applies to)
   - **Formatting required**: "Y" if specific formatting/symbols required, "N/A" otherwise
   - **Required in Print**: "y" if must be in physical manual, "n" if digital is OK, "N/A" if unclear
   - **Comments**: Any special notes (e.g., "inside instructions section", "warning token", etc.)

5. If you find numbered subsections (like 7.1, 7.6, 7.12), split them into SEPARATE requirements
6. If you find lettered items (like a), b), c)), split them into SEPARATE requirements
7. Preserve ALL specific details: measurements, voltages, symbols, legal keywords (shall, must)

Respond with a JSON object:
{{
  "requirements": [
    {{
      "Description": "Full requirement text here",
      "Clause/Requirement": "7.1",
      "Requirement scope": "battery, charger",
      "Formatting required?": "Y",
      "Required in Print?": "y",
      "Comments": "inside instructions section"
    }},
    ...
  ],
  "extraction_notes": "Any important observations about the extraction",
  "confidence": "high|medium|low"
}}

Extract ALL relevant requirements. Be thorough but precise."""

    try:
        print(f"[AI EXTRACTION] Sending {len(pdf_text)} chars to Claude Opus...")

        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=16000,
            temperature=0,  # Deterministic for extraction
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        response_text = message.content[0].text
        print(f"[AI EXTRACTION] Received response: {len(response_text)} chars")

        # Parse JSON response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)

        # Add standard name to each requirement
        requirements = result.get('requirements', [])
        for req in requirements:
            req['Standard/Reg'] = standard_name or 'Unknown'

        print(f"[AI EXTRACTION] Extracted {len(requirements)} requirements")

        return {
            'rows': requirements,
            'stats': {
                'total_detected': len(requirements),
                'classified_rows': len(requirements),
                'extraction_notes': result.get('extraction_notes', '')
            },
            'confidence': result.get('confidence', 'medium')
        }

    except anthropic.APIError as e:
        print(f"[AI EXTRACTION ERROR] API error: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Claude API error: {str(e)}")

    except json.JSONDecodeError as e:
        print(f"[AI EXTRACTION ERROR] JSON parse error: {e}")
        print(f"Response was: {response_text[:500]}")
        raise ValueError(f"Failed to parse AI response as JSON: {str(e)}")

    except Exception as e:
        print(f"[AI EXTRACTION ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"AI extraction failed: {str(e)}")


def validate_requirement_schema(req: Dict) -> Dict:
    """
    Validate and normalize a requirement dict to match our 7-column schema.
    """
    schema_columns = [
        'Description',
        'Standard/Reg',
        'Clause/Requirement',
        'Requirement scope',
        'Formatting required?',
        'Required in Print?',
        'Comments'
    ]

    normalized = {}
    for col in schema_columns:
        normalized[col] = req.get(col, 'N/A')

    return normalized
