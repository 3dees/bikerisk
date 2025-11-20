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
from extract_gpt import extract_requirements_gpt
from detect import detect_manual_sections, detect_all_sections, merge_small_sections
from classify import rows_to_csv_dicts
from validate import parse_parent_section

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
    import time
    start_time = time.time()

    try:
        # Step 1: Extract text (always needed for both modes)
        with results_lock:
            RESULTS_STORE[job_id].update({
                'status': 'processing',
                'step': 'Extracting text from PDF',
                'progress': 10,
                'start_time': start_time
            })

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
        if extraction_mode == "gpt":
            # GPT-4o-mini MODE: Direct extraction, no section detection
            print(f"[EXTRACTION] Using GPT-4o-mini mode (direct extraction)")

            # Step 2: Extract full text
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'step': 'Preparing text for GPT extraction',
                    'progress': 20
                })

            pdf_text = extraction_result.get('text', '')
            if not pdf_text:
                blocks = extraction_result['blocks']
                pdf_text = '\n'.join([block['raw'] for block in blocks])

            # Step 3: GPT Extraction with progress callback
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'step': 'Extracting requirements with GPT-4o-mini',
                    'progress': 30
                })

            def update_gpt_progress(completed_chunks, total_chunks, requirements_count):
                progress = 30 + int((completed_chunks / total_chunks) * 50)  # 30-80% range
                with results_lock:
                    RESULTS_STORE[job_id].update({
                        'step': f'Processing chunk {completed_chunks}/{total_chunks}',
                        'progress': progress,
                        'requirements_extracted': requirements_count
                    })

            # Extract with GPT (clause + text only)
            raw_requirements = extract_requirements_gpt(
                pdf_text=pdf_text,
                filename=standard_name or filename,
                extraction_type=extraction_type,
                api_key=api_key,
                max_workers=5,
                progress_callback=update_gpt_progress
            )

            # Step 4: Classify requirements using existing classify.py
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'step': 'Classifying requirements',
                    'progress': 85
                })

            # Convert GPT format to classified format
            # GPT returns: [{"clause": "X", "text": "Y"}]
            # We need: [{"Clause/Requirement": "X", "Description": "Y", "Standard/Reg": ...}]
            classified_rows = []
            for req in raw_requirements:
                clause = req.get('clause', '')
                parent = parse_parent_section(clause) or 'Unknown'

                classified_rows.append({
                    'Clause/Requirement': clause,
                    'Description': req.get('text', ''),
                    'Standard/Reg': standard_name or 'Unknown',
                    'Requirement scope': '',  # Will be filled by classify.py if needed
                    'Formatting required?': 'N/A',
                    'Required in Print?': 'n',
                    'Parent Section': parent,  # Parsed from clause number
                    'Sub-section': 'N/A',
                    'Comments': '',  # Leave empty per user guidance
                    'Contains Image?': 'N',
                    'Safety Notice Type': 'None'
                })

            # Convert to CSV format
            csv_rows = rows_to_csv_dicts(classified_rows)

            stats = {
                'total_detected': len(classified_rows),
                'classified_rows': len(classified_rows),
                'extraction_method': 'gpt-4o-mini'
            }
            confidence = 'high'
            consolidations = []
            extraction_method = "gpt-4o-mini"

            elapsed_time = time.time() - start_time

            # Store completed results
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'status': 'completed',
                    'step': 'Complete',
                    'extraction_method': extraction_method,
                    'extraction_confidence': confidence,
                    'rows': classified_rows,
                    'csv_rows': csv_rows,
                    'consolidations': consolidations,
                    'stats': stats,
                    'progress': 100,
                    'elapsed_time': round(elapsed_time, 2)
                })

        elif extraction_mode == "ai":
            # HYBRID AI MODE: Rules find sections, AI extracts requirements
            print(f"[EXTRACTION] Using HYBRID AI mode (rules + Claude Opus)")

            # Step 2: Detect sections
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'step': 'Detecting sections',
                    'progress': 20
                })

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
                with results_lock:
                    RESULTS_STORE[job_id].update({
                        'step': 'No sections found, using full document extraction',
                        'progress': 30
                    })

                print(f"[EXTRACTION] No sections found, trying full AI extraction as fallback")
                pdf_text = extraction_result.get('text', '')
                if not pdf_text:
                    blocks = extraction_result['blocks']
                    pdf_text = '\n'.join([block['raw'] for block in blocks])

                ai_result = extract_requirements_with_ai(pdf_text, standard_name, extraction_type, api_key)
            else:
                # Step 3: AI Extraction with progress updates
                with results_lock:
                    RESULTS_STORE[job_id].update({
                        'step': f'Extracting requirements from {len(sections)} sections',
                        'progress': 40
                    })

                # Step 2: Pass detected sections to AI for intelligent extraction (clause-level batching)
                # Using optimal batch size of 30 clauses to stay under 5-minute timeout
                # (Phase 1 clause segmentation = 6-9x speed improvement)
                clauses_per_batch = 30

                print(f"[EXTRACTION] Processing {len(sections)} sections with clause-level batching (30 clauses/batch)")

                # Create a progress callback
                def update_extraction_progress(completed_batches, total_batches, requirements_count):
                    progress = 40 + int((completed_batches / total_batches) * 40)  # 40-80% range
                    with results_lock:
                        RESULTS_STORE[job_id].update({
                            'step': f'Extracting batch {completed_batches}/{total_batches}',
                            'progress': progress,
                            'requirements_extracted': requirements_count
                        })

                ai_result = extract_from_detected_sections_batched(
                    sections,
                    standard_name,
                    extraction_type,
                    api_key,
                    clauses_per_batch=clauses_per_batch,
                    progress_callback=update_extraction_progress,
                    job_id=job_id
                )

            # Step 4: Finalization
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'step': 'Finalizing results',
                    'progress': 90
                })

            classified_rows = ai_result['rows']
            stats = ai_result['stats']
            confidence = ai_result['confidence']
            consolidations = []  # Consolidation happens separately in the frontend (Tab 2)

            # Convert to CSV-friendly format
            csv_rows = rows_to_csv_dicts(classified_rows)

            extraction_method = "hybrid_ai_rules"

            elapsed_time = time.time() - start_time

            # Store completed results
            with results_lock:
                RESULTS_STORE[job_id].update({
                    'status': 'completed',
                    'step': 'Complete',
                    'extraction_method': extraction_method,
                    'extraction_confidence': confidence,
                    'rows': classified_rows,
                    'csv_rows': csv_rows,
                    'consolidations': consolidations,
                    'stats': stats,
                    'progress': 100,
                    'elapsed_time': round(elapsed_time, 2)
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
    extraction_mode: Optional[str] = "gpt",  # "gpt", "ai", or "rules"
    extraction_type: Optional[str] = "manual",  # "manual" or "all"
    authorization: Optional[str] = Header(None)
):
    """
    Upload a PDF and start background extraction. Returns immediately with job_id.

    Args:
        file: PDF file
        standard_name: Optional name of the standard (e.g., "EN 15194")
        custom_section_name: Optional custom section name to search for (e.g., "Instruction for use")
        extraction_mode: "gpt" (default, GPT-4o-mini), "ai" (Claude hybrid), or "rules" (regex only)
        extraction_type: "manual" (default) for manual requirements only, "all" for all requirements
        authorization: Authorization header with Bearer token (OpenAI API key for gpt mode, Anthropic for ai mode)

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
        if extraction_mode == "gpt":
            api_key = os.getenv('OPENAI_API_KEY')
        else:  # "ai" mode
            api_key = os.getenv('ANTHROPIC_API_KEY')

    # Validate API key based on extraction mode
    if extraction_mode == "gpt" and not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key required for GPT extraction mode. Please provide it via Authorization header or set OPENAI_API_KEY env var."
        )
    elif extraction_mode == "ai" and not api_key:
        raise HTTPException(
            status_code=400,
            detail="Anthropic API key required for AI extraction mode. Please provide it via Authorization header or set ANTHROPIC_API_KEY env var."
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
