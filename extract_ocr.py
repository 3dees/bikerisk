"""
OCR extraction for image-based PDFs using Google Cloud Vision API.
Fallback to Claude PDF Vision if Google Cloud is unavailable.
"""
import os
import io
import base64
from typing import Dict, Tuple
from dotenv import load_dotenv
import anthropic
import httpx

load_dotenv()


def extract_text_with_google_vision(file_bytes: bytes, filename: str = "document.pdf") -> Tuple[str, str, bool]:
    """
    Extract text from image-based PDF using Google Cloud Vision API.

    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename for logging

    Returns:
        Tuple of (extracted_text, method, success)
    """
    try:
        from google.cloud import vision

        # Check for credentials
        credentials_json = os.getenv('GOOGLE_CLOUD_VISION_CREDENTIALS')
        if not credentials_json:
            print(f"[OCR] Google Cloud Vision credentials not found, skipping")
            return "", "google_vision_skipped", False

        # Set credentials from environment variable
        if credentials_json:
            import json
            import tempfile

            # Write credentials to temp file for Google client
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(json.loads(credentials_json), f)
                temp_cred_file = f.name

            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_cred_file

        print(f"[OCR] Using Google Cloud Vision API for {filename}")

        # Initialize the client
        client = vision.ImageAnnotatorClient()

        # For multi-page PDFs, we need to use pdf2image to convert to images first
        try:
            from pdf2image import convert_from_bytes

            # Convert PDF to images (one per page)
            print(f"[OCR] Converting PDF to images (this may take a moment)...")
            try:
                images = convert_from_bytes(file_bytes, dpi=200)
                print(f"[OCR] Successfully converted to {len(images)} images")
            except Exception as conversion_error:
                print(f"[OCR] PDF to image conversion failed: {str(conversion_error)}")
                print(f"[OCR] This usually means poppler-utils is not installed")
                raise

            print(f"[OCR] Processing {len(images)} pages with Google Vision...")

            all_text = []
            for i, img in enumerate(images, 1):
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                # Process with Google Vision
                image = vision.Image(content=img_bytes)
                response = client.document_text_detection(image=image)

                if response.error.message:
                    print(f"[OCR] Warning: Page {i} error: {response.error.message}")
                    continue

                # Extract text from this page
                if response.full_text_annotation:
                    all_text.append(response.full_text_annotation.text)
                    print(f"[OCR] Page {i}/{len(images)}: {len(response.full_text_annotation.text)} chars")

        except ImportError:
            # pdf2image not available, try direct PDF processing (may only work for single page)
            print(f"[OCR] pdf2image not available, trying direct PDF processing (may fail for multi-page)")
            image = vision.Image(content=file_bytes)
            response = client.document_text_detection(image=image)

            if response.error.message:
                raise Exception(f"Google Vision API error: {response.error.message}")

            all_text = []
            if response.full_text_annotation:
                all_text.append(response.full_text_annotation.text)

        # Cleanup temp credentials file
        if credentials_json and os.path.exists(temp_cred_file):
            os.remove(temp_cred_file)

        extracted_text = '\n\n'.join(all_text)

        if extracted_text and len(extracted_text.strip()) > 100:
            print(f"[OCR] Google Vision extracted {len(extracted_text)} characters")
            return extracted_text, "google_vision", True
        else:
            print(f"[OCR] Google Vision extracted insufficient text")
            return extracted_text, "google_vision_insufficient", False

    except ImportError as e:
        print(f"[OCR] Missing dependency: {str(e)}")
        return "", "google_vision_missing_dependency", False
    except Exception as e:
        print(f"[OCR] Google Vision failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", f"google_vision_error", False


def extract_text_with_claude_vision(file_bytes: bytes, filename: str = "document.pdf", api_key: str = None) -> Tuple[str, str, bool]:
    """
    Extract text from PDF using Claude's native PDF vision capabilities.

    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename for logging
        api_key: Anthropic API key

    Returns:
        Tuple of (extracted_text, method, success)
    """
    try:
        if not api_key:
            api_key = os.getenv('ANTHROPIC_API_KEY')

        if not api_key:
            print(f"[OCR] No Anthropic API key available")
            return "", "claude_vision_no_key", False

        print(f"[OCR] Using Claude PDF Vision for {filename}")

        # Setup client
        no_proxy = os.getenv('NO_PROXY', '')
        if 'anthropic.com' not in no_proxy:
            os.environ['NO_PROXY'] = no_proxy + ',anthropic.com,*.anthropic.com' if no_proxy else 'anthropic.com,*.anthropic.com'

        try:
            http_client = httpx.Client(timeout=120.0)
            client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        except Exception:
            client = anthropic.Anthropic(api_key=api_key)

        # Encode PDF as base64
        pdf_base64 = base64.standard_b64encode(file_bytes).decode('utf-8')

        # Send PDF to Claude for text extraction
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=16000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": """Extract all text content from this PDF document.

Preserve the original structure and formatting as much as possible.
Include all visible text, headers, footers, page numbers, and content.
Maintain paragraph breaks and section divisions.

Return ONLY the extracted text, no additional commentary."""
                    }
                ]
            }]
        )

        extracted_text = message.content[0].text if message.content else ""

        if extracted_text and len(extracted_text.strip()) > 100:
            print(f"[OCR] Claude Vision extracted {len(extracted_text)} characters")
            return extracted_text, "claude_vision", True
        else:
            print(f"[OCR] Claude Vision extracted insufficient text")
            return extracted_text, "claude_vision_insufficient", False

    except Exception as e:
        print(f"[OCR] Claude Vision failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", "claude_vision_error", False


def extract_with_ocr(file_bytes: bytes, filename: str = "document.pdf") -> Dict:
    """
    Extract text from image-based PDF using OCR.
    Tries Google Cloud Vision first, then falls back to Claude PDF Vision.

    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename for logging

    Returns:
        Dict with keys:
        - 'raw_text': extracted text
        - 'method': extraction method used
        - 'success': bool
        - 'error': error message if failed
        - 'confidence': extraction confidence
    """
    print(f"[OCR] Starting OCR extraction for {filename}")

    # Try Google Cloud Vision first
    text, method, success = extract_text_with_google_vision(file_bytes, filename)

    if success and text:
        return {
            'raw_text': text,
            'method': method,
            'success': True,
            'error': None,
            'confidence': 'high'
        }

    # Fallback to Claude PDF Vision
    print(f"[OCR] Google Vision unavailable or failed, trying Claude PDF Vision")
    text, method, success = extract_text_with_claude_vision(file_bytes, filename)

    if success and text:
        return {
            'raw_text': text,
            'method': method,
            'success': True,
            'error': None,
            'confidence': 'medium'
        }

    # Both failed
    return {
        'raw_text': '',
        'method': 'ocr_failed',
        'success': False,
        'error': f'OCR extraction failed. Google Vision and Claude Vision both unable to extract text from {filename}.',
        'confidence': 'low'
    }
