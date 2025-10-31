"""
FastAPI backend for e-bike standards requirement extraction.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from extract import extract_from_file
from extract_ai import extract_requirements_with_ai, extract_from_detected_sections
from detect import (
    detect_manual_sections,
    detect_manual_clauses,
    combine_detections
)
from classify import classify_detected_items, rows_to_csv_dicts
from consolidate import consolidate_requirements

load_dotenv()


# In-memory storage for results
RESULTS_STORE: Dict[str, Dict] = {}

app = FastAPI(
    title="E-Bike Standards Requirement Extractor",
    description="Extract instruction/manual requirements from e-bike standards and regulations",
    version="0.1.0"
)

# Add CORS middleware to allow Streamlit to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "E-Bike Standards Requirement Extractor API",
        "version": "0.1.0"
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    standard_name: Optional[str] = None,
    custom_section_name: Optional[str] = None,
    extraction_mode: Optional[str] = "ai",  # "ai" or "rules"
    api_key: Optional[str] = None
):
    """
    Upload a PDF and extract manual/instruction requirements.

    Args:
        file: PDF file
        standard_name: Optional name of the standard (e.g., "EN 15194")
        custom_section_name: Optional custom section name to search for (e.g., "Instruction for use")
        extraction_mode: "ai" (default) or "rules" for extraction method
        api_key: Anthropic API key (uses env var if not provided)

    Returns:
        Job ID and initial results
    """
    # Generate job ID
    job_id = str(uuid.uuid4())

    # Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Use filename as standard name if not provided
    if not standard_name:
        standard_name = file.filename or "Unknown Standard"

    # Get API key from env if not provided
    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    # Step 1: Extract text (always needed for both modes)
    extraction_result = extract_from_file(file_bytes, file.filename)

    if not extraction_result['success']:
        # Partial failure - return warning
        RESULTS_STORE[job_id] = {
            'job_id': job_id,
            'filename': file.filename,
            'standard_name': standard_name,
            'status': 'failed',
            'error': extraction_result['error'],
            'extraction_method': extraction_result['method'],
            'rows': [],
            'consolidations': [],
            'created_at': datetime.now().isoformat()
        }
        raise HTTPException(
            status_code=422,
            detail={
                'message': extraction_result['error'],
                'suggestion': 'Please try converting the PDF to a different format or ensure it is not image-based.',
                'job_id': job_id
            }
        )

    # Branch based on extraction mode
    if extraction_mode == "ai":
        # HYBRID AI MODE: Rules find sections, AI extracts requirements
        print(f"[EXTRACTION] Using HYBRID AI mode (rules + Claude Opus)")

        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="API key required for AI extraction mode. Please provide it or set ANTHROPIC_API_KEY env var."
            )

        try:
            # Step 1: Use rule-based detection to find ALL relevant sections (scans entire document)
            blocks = extraction_result['blocks']
            custom_names = [custom_section_name] if custom_section_name else None
            manual_sections = detect_manual_sections(blocks, custom_names)

            print(f"[HYBRID AI] Rules found {len(manual_sections)} manual sections")

            if not manual_sections:
                # No sections found by rules - fallback to full AI extraction
                print(f"[HYBRID AI] No sections found by rules, trying full AI extraction as fallback")
                pdf_text = extraction_result.get('text', '')
                if not pdf_text:
                    blocks = extraction_result['blocks']
                    pdf_text = '\n'.join([block['raw'] for block in blocks])

                ai_result = extract_requirements_with_ai(pdf_text, standard_name, api_key)
            else:
                # Step 2: Pass detected sections to AI for intelligent extraction
                ai_result = extract_from_detected_sections(manual_sections, standard_name, api_key)

            classified_rows = ai_result['rows']
            stats = ai_result['stats']
            confidence = ai_result['confidence']
            consolidations = []  # TODO: Add AI consolidation later

            # Convert to CSV-friendly format
            csv_rows = rows_to_csv_dicts(classified_rows)

            extraction_method = "hybrid_ai_rules"

        except Exception as e:
            print(f"[HYBRID AI EXTRACTION FAILED] {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"AI extraction failed: {str(e)}. Try rule-based mode instead."
            )

    else:
        # RULE-BASED MODE: Original logic
        print(f"[EXTRACTION] Using rule-based mode")

        # Step 2: Detect manual sections (Pass A)
        blocks = extraction_result['blocks']
        custom_names = [custom_section_name] if custom_section_name else None
        manual_sections = detect_manual_sections(blocks, custom_names)

        # Step 3: Detect manual clauses (Pass B)
        manual_clauses = detect_manual_clauses(blocks, manual_sections)

        # Step 4: Combine detections
        detected_items = combine_detections(manual_sections, manual_clauses)

        # Step 5: Classify into schema
        classified_rows = classify_detected_items(detected_items, standard_name)

        # Step 6: Consolidate (Phase 3 - placeholder for now)
        consolidations = consolidate_requirements(classified_rows)

        # Convert to CSV-friendly format
        csv_rows = rows_to_csv_dicts(classified_rows)

        stats = {
            'total_detected': len(detected_items),
            'manual_sections': len(manual_sections),
            'manual_clauses': len(manual_clauses),
            'classified_rows': len(classified_rows),
        }
        confidence = extraction_result['confidence']
        extraction_method = extraction_result['method']

    # Store results
    RESULTS_STORE[job_id] = {
        'job_id': job_id,
        'filename': file.filename,
        'standard_name': standard_name,
        'status': 'completed',
        'extraction_method': extraction_method,
        'extraction_confidence': confidence,
        'extraction_mode': extraction_mode,  # NEW: track which mode was used
        'rows': classified_rows,  # Keep internal fields for UI
        'csv_rows': csv_rows,     # Clean version for export
        'consolidations': consolidations,
        'stats': stats,
        'created_at': datetime.now().isoformat()
    }

    return {
        'job_id': job_id,
        'status': 'completed',
        'filename': file.filename,
        'standard_name': standard_name,
        'extraction_method': extraction_method,
        'extraction_confidence': confidence,
        'extraction_mode': extraction_mode,
        'stats': stats
    }


@app.get("/results/{job_id}")
def get_results(job_id: str):
    """
    Get extraction results for a job.

    Args:
        job_id: Job ID from upload

    Returns:
        Full results including rows and consolidations
    """
    if job_id not in RESULTS_STORE:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")

    return RESULTS_STORE[job_id]


@app.get("/export/{job_id}")
def export_csv(job_id: str):
    """
    Export results as CSV.

    Args:
        job_id: Job ID from upload

    Returns:
        CSV-formatted results
    """
    if job_id not in RESULTS_STORE:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")

    result = RESULTS_STORE[job_id]
    csv_rows = result['csv_rows']

    if not csv_rows:
        return JSONResponse(
            content={'message': 'No results to export'},
            status_code=204
        )

    # Convert to CSV format
    import io
    import csv

    output = io.StringIO()
    fieldnames = [
        'Description',
        'Standard/Reg',
        'Clause/Requirement',
        'Requirement scope',
        'Formatting required?',
        'Required in Print?',
        'Comments'
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

    csv_content = output.getvalue()

    return JSONResponse(
        content={'csv': csv_content},
        headers={
            'Content-Disposition': f'attachment; filename="{result["filename"]}_requirements.csv"'
        }
    )


@app.delete("/results/{job_id}")
def delete_results(job_id: str):
    """
    Delete results for a job.

    Args:
        job_id: Job ID to delete

    Returns:
        Success message
    """
    if job_id not in RESULTS_STORE:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")

    del RESULTS_STORE[job_id]
    return {'message': f'Results for job {job_id} deleted'}


@app.get("/jobs")
def list_jobs():
    """
    List all jobs.

    Returns:
        List of job summaries
    """
    jobs = []
    for job_id, result in RESULTS_STORE.items():
        jobs.append({
            'job_id': job_id,
            'filename': result['filename'],
            'standard_name': result['standard_name'],
            'status': result['status'],
            'created_at': result['created_at'],
            'num_rows': len(result['rows'])
        })

    return {'jobs': jobs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
