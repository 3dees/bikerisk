"""
Test GPT extraction on SS_EN_50604 - the problem child.

This standard has unusual "Replacement:" formatting that regex misses.
Expected: >350 requirements (vs 136 with regex gate)
"""
import sys
from extract import extract_from_file
from extract_gpt import extract_requirements_gpt
import json


def test_ss_en_extraction():
    """Test GPT extraction on SS_EN_50604."""

    pdf_path = r"files\SS_EN_50604_1_EN.pdf"

    print("=" * 80)
    print("TEST: GPT Extraction on SS_EN_50604 (The Problem Child)")
    print("=" * 80)
    print("\nThis standard uses 'Replacement:' format that causes data loss with regex.")
    print("Current regex approach finds: ~136 requirements")
    print("Expected with GPT: >350 requirements")
    print()

    # Read PDF
    print(f"[1] Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    # Extract text using extract_from_file
    print("[2] Extracting text from PDF...")
    extraction_result = extract_from_file(pdf_bytes, pdf_path)

    if not extraction_result['success']:
        print(f"[ERROR] Text extraction failed: {extraction_result.get('error', 'Unknown error')}")
        return None

    # Get full text
    full_text = extraction_result.get('text', '')
    if not full_text:
        blocks = extraction_result['blocks']
        full_text = '\n'.join([block['raw'] for block in blocks])

    print(f"[3] Full text length: {len(full_text):,} characters")

    # Run GPT extraction
    print("\n[4] Running GPT extraction (this may take 2-3 minutes)...")
    print("-" * 80)

    try:
        requirements = extract_requirements_gpt(
            pdf_text=full_text,
            filename="SS_EN_50604",
            extraction_type="all",
            max_workers=5
        )

        print("-" * 80)
        print(f"\n[5] RESULTS:")
        print(f"    Total requirements extracted: {len(requirements)}")
        print()

        # Check for "Replacement:" sections (the problematic format)
        replacement_reqs = [r for r in requirements if 'replacement' in r.get('text', '').lower()]
        print(f"    Requirements with 'Replacement:' format: {len(replacement_reqs)}")

        # Check for specific clause ranges we expect
        clause_5_reqs = [r for r in requirements if r.get('clause', '').startswith('5.1.10')]
        print(f"    Requirements in 5.1.10x range: {len(clause_5_reqs)}")

        # Show sample requirements
        print(f"\n[6] Sample requirements (first 10):")
        for i, req in enumerate(requirements[:10], 1):
            clause = req.get('clause', 'N/A')
            text = req.get('text', '')
            text_preview = text[:80] + "..." if len(text) > 80 else text
            print(f"    {i}. [{clause}] {text_preview}")

        # Save results for inspection
        output_path = "test_ss_en_50604_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(requirements, f, indent=2, ensure_ascii=False)

        print(f"\n[7] Full results saved to: {output_path}")

        # Success criteria
        print(f"\n[8] Quality Check:")

        if len(requirements) > 350:
            print(f"    SUCCESS: Found {len(requirements)} requirements (expected >350)")
            print(f"    This is {len(requirements) - 136} more than regex approach!")
        elif len(requirements) > 250:
            print(f"    PARTIAL: Found {len(requirements)} requirements (expected >350)")
            print(f"    Better than regex (136) but below target")
        else:
            print(f"    WARNING: Only found {len(requirements)} requirements (expected >350)")

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)

        return requirements

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_ss_en_extraction()

    if results and len(results) > 350:
        print(f"\nSUCCESS: Test passed - extracted {len(results)} requirements")
        sys.exit(0)
    elif results:
        print(f"\nPARTIAL: Found {len(results)} requirements (target: 350)")
        sys.exit(0)
    else:
        print(f"\nFAILED: Test failed")
        sys.exit(1)
