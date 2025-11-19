"""
FastAPI backend for e-bike standards requirement extraction.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import anthropic
from concurrent.futures import ThreadPoolExecutor
import threading

from extract import extract_from_file
from extract_ai import extract_requirements_with_ai, extract_from_detected_sections, extract_from_detected_sections_batched
from detect import detect_manual_sections, detect_all_sections, merge_small_sections
from classify import rows_to_csv_dicts

load_dotenv()


# In-memory storage for results
RESULTS_STORE: Dict[str, Dict] = {}

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=2)

# Lock for thread-safe RESULTS_STORE updates
results_lock = threading.Lock()

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


def _process_upload_background(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    standard_name: str,
    custom_section_name: Optional[str],
    extraction_mode: str,
    extraction_type: str,
    api_key: Optional[str]
):
    """
    Background processing function for PDF extraction.
    Updates RESULTS_STORE with progress and final results.
    """
    try:
        # Step 1: Extract text (always needed for both modes)
        extraction_result = extract_from_file(file_bytes, filename)

        if not extraction_result['success']:
            # Partial failure - store error
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'status': 'failed',
                    'error': extraction_result['error'],
                    'extraction_method': extraction_result['method']
                })
            return

        # Branch based on extraction mode
        if extraction_mode == "ai":
            # HYBRID AI MODE: Rules find sections, AI extracts requirements
            print(f"[EXTRACTION] Using HYBRID AI mode (rules + Claude Opus)")

            # Step 1: Use appropriate detection based on extraction_type
            blocks = extraction_result['blocks']
            custom_names = [custom_section_name] if custom_section_name else None

            if extraction_type == "all":
                # All Requirements mode: detect all numbered sections
                sections = detect_all_sections(blocks, custom_names)
                print(f"[ALL REQUIREMENTS] Detected {len(sections)} sections")

                # Merge small sections to reduce processing time
                sections = merge_small_sections(sections)
            else:
                # Manual Requirements mode: detect manual-specific sections
                sections = detect_manual_sections(blocks, custom_names)
                print(f"[MANUAL REQUIREMENTS] Detected {len(sections)} manual sections")

                # Also merge manual sections for speed
                sections = merge_small_sections(sections)

            if not sections:
                # No sections found by rules - fallback to full AI extraction
                print(f"[EXTRACTION] No sections found, trying full AI extraction as fallback")
                pdf_text = extraction_result.get('text', '')
                if not pdf_text:
                    blocks = extraction_result['blocks']
                    pdf_text = '\n'.join([block['raw'] for block in blocks])

                ai_result = extract_requirements_with_ai(pdf_text, standard_name, extraction_type, api_key)
            else:
                # Step 2: Pass detected sections to AI for intelligent extraction (batched for performance)
                # Auto-adjust batch size based on document size
                if len(sections) > 100:
                    batch_size = 3  # Large docs: small batches for reliability
                elif len(sections) > 50:
                    batch_size = 5  # Medium docs: balanced
                else:
                    batch_size = 10  # Small docs: larger batches for speed

                print(f"[EXTRACTION] Using batch_size={batch_size} for {len(sections)} sections")

                ai_result = extract_from_detected_sections_batched(
                    sections,
                    standard_name,
                    extraction_type,
                    api_key,
                    batch_size=batch_size
                )

            classified_rows = ai_result['rows']
            stats = ai_result['stats']
            confidence = ai_result['confidence']
            consolidations = []  # Consolidation happens separately in the frontend (Tab 2)

            # Convert to CSV-friendly format
            csv_rows = rows_to_csv_dicts(classified_rows)

            extraction_method = "hybrid_ai_rules"

            # Store completed results
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'status': 'completed',
                    'extraction_method': extraction_method,
                    'extraction_confidence': confidence,
                    'rows': classified_rows,
                    'csv_rows': csv_rows,
                    'consolidations': consolidations,
                    'stats': stats,
                    'progress': 100
                })

    except anthropic.APIStatusError as e:
        # Anthropic API error (overload, rate limit, etc)
        print(f"[API ERROR] Anthropic API error: {e}")
        with results_lock:
            RESULTS_STORE[job_id].update({
                'status': 'failed',
                'error': f"Anthropic API temporarily unavailable: {str(e)}. Please try again in a few moments."
            })

    except anthropic.APIError as e:
        # Other Anthropic errors
        print(f"[API ERROR] Anthropic error: {e}")
        with results_lock:
            RESULTS_STORE[job_id].update({
                'status': 'failed',
                'error': f"AI service error: {str(e)}"
            })

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"[EXTRACTION ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        with results_lock:
            RESULTS_STORE[job_id].update({
                'status': 'failed',
                'error': f"Extraction failed: {str(e)}"
            })


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    standard_name: Optional[str] = None,
    custom_section_name: Optional[str] = None,
    extraction_mode: Optional[str] = "ai",  # "ai" or "rules"
    extraction_type: Optional[str] = "manual",  # "manual" or "all"
    authorization: Optional[str] = Header(None)
):
    """
    Upload a PDF and start background extraction. Returns immediately with job_id.

    Args:
        file: PDF file
        standard_name: Optional name of the standard (e.g., "EN 15194")
        custom_section_name: Optional custom section name to search for (e.g., "Instruction for use")
        extraction_mode: "ai" (default) or "rules" for extraction method
        extraction_type: "manual" (default) for manual requirements only, "all" for all requirements
        authorization: Authorization header with Bearer token (API key)

    Returns:
        Job ID and status='processing' (use /status/{job_id} to poll for completion)
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

    # Extract API key from Authorization header
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")

    # Fallback to env var if no header provided
    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')

    # Validate API key for AI mode
    if extraction_mode == "ai" and not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key required for AI extraction mode. Please provide it via Authorization header or set ANTHROPIC_API_KEY env var."
        )

    # Initialize job in RESULTS_STORE with status='processing'
    with results_lock:
        RESULTS_STORE[job_id] = {
            'job_id': job_id,
            'filename': file.filename,
            'standard_name': standard_name,
            'status': 'processing',
            'progress': 0,
            'extraction_mode': extraction_mode,
            'extraction_type': extraction_type,
            'created_at': datetime.now().isoformat()
        }

    # Submit background task
    executor.submit(
        _process_upload_background,
        job_id,
        file_bytes,
        file.filename,
        standard_name,
        custom_section_name,
        extraction_mode,
        extraction_type,
        api_key
    )

    print(f"[UPLOAD] Job {job_id} submitted for background processing")

    # Return immediately with job_id and status='processing'
    return {
        'job_id': job_id,
        'status': 'processing',
        'filename': file.filename,
        'standard_name': standard_name,
        'message': 'Processing started. Use /status/{job_id} to check progress.'
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """
    Get processing status for a job (lightweight endpoint for polling).

    Args:
        job_id: Job ID from upload

    Returns:
        Status information: {status, progress, filename, error (if failed)}
    """
    if job_id not in RESULTS_STORE:
        return {"status": "not_found", "job_id": job_id}

    job = RESULTS_STORE[job_id]
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "filename": job.get("filename"),
        "error": job.get("error")
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
