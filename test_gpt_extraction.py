"""
Test GPT extraction on IEC 62133-2 standard.
"""
import sys
from extract import extract_from_file
from extract_gpt import extract_requirements_gpt
import json


def test_iec_62133_extraction():
    """Test GPT extraction on IEC 62133-2."""

    pdf_path = r"files\IEC 62133-2 {ed1.0}.pdf"

    print("=" * 80)
    print("TEST: GPT Extraction on IEC 62133-2")
    print("=" * 80)

    # Read PDF
    print(f"\n[1] Reading PDF: {pdf_path}")
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
            filename="IEC 62133-2",
            extraction_type="all",  # Extract everything for comparison
            max_workers=5
        )

        print("-" * 80)
        print(f"\n[5] RESULTS:")
        print(f"    Total requirements extracted: {len(requirements)}")

        # Show sample requirements
        print(f"\n[6] Sample requirements (first 10):")
        for i, req in enumerate(requirements[:10], 1):
            clause = req.get('clause', 'N/A')
            text = req.get('text', '')
            text_preview = text[:80] + "..." if len(text) > 80 else text
            print(f"    {i}. [{clause}] {text_preview}")

        # Check for specific clauses we expect
        print(f"\n[7] Verification:")
        clause_ids = {req['clause'] for req in requirements if req.get('clause')}

        expected_clauses = ['7.2.1.a', '7.3.2', '9.1', '5.2']
        found_expected = [c for c in expected_clauses if c in clause_ids]

        print(f"    Expected clauses found: {len(found_expected)}/{len(expected_clauses)}")
        for clause in expected_clauses:
            status = "✓" if clause in clause_ids else "✗"
            print(f"      {status} {clause}")

        # Save results for inspection
        output_path = "test_gpt_extraction_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(requirements, f, indent=2, ensure_ascii=False)

        print(f"\n[8] Full results saved to: {output_path}")

        # Success criteria
        print(f"\n[9] Quality Check:")
        if len(requirements) > 300:
            print(f"    ✓ Found {len(requirements)} requirements (expected >300)")
        else:
            print(f"    ✗ Only found {len(requirements)} requirements (expected >300)")

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
    results = test_iec_62133_extraction()

    if results:
        print(f"\n✅ Test passed - extracted {len(results)} requirements")
        sys.exit(0)
    else:
        print(f"\n✗ Test failed")
        sys.exit(1)
