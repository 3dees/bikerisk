# BikeRisk - E-Bike Standards Requirement Extractor

## Architecture Overview

**Two-tier web application** extracting instruction manual requirements from e-bike safety standards PDFs:
- **Backend:** FastAPI (`main.py`) with in-memory job storage (`RESULTS_STORE` dict)
- **Frontend:** Streamlit (`app.py`) with dual-tab interface
- **Core Pipeline:** `extract.py` → `detect.py` → `classify.py` → `consolidate_*.py`

### Extraction Modes

**Hybrid AI Mode (preferred):** Rules detect sections (`detect_manual_sections`) → Claude extracts requirements (`extract_from_detected_sections` in `extract_ai.py`)
- Rule-based detection scans entire document for manual/instruction sections
- AI processes only detected sections (avoids token limits, faster, cheaper)
- Fallback to full AI extraction if no sections found

**Rule-Based Mode:** Pure regex patterns for section detection, clause identification, and classification
- Uses patterns like `MANUAL_SECTION_PATTERNS`, `CLAUSE_KEYWORD_PATTERNS` in `detect.py`
- Two-pass detection: dedicated sections (Pass A) + scattered clauses (Pass B)

## Critical Workflows

### Starting Development Environment
```bash
# Windows (launches both servers in separate windows)
start.bat

# Linux/Mac
./start.sh

# Manual start (two terminals required)
python main.py              # FastAPI on :8000
streamlit run app.py        # Streamlit on :8501
```

**Note:** FastAPI must start first; Streamlit calls API endpoints like `/upload`, `/results/{job_id}`

### API Key Management
- Anthropic API key required for AI modes
- Load from `.env` file (`ANTHROPIC_API_KEY=sk-ant-...`)
- Streamlit sidebar allows session override
- Proxy bypass pattern: `os.environ['NO_PROXY'] = '*.anthropic.com'` (see `extract_ai.py:47`)

## Data Schema & Processing

### 8-Column Requirement Schema
All requirements follow this structure (see `classify.py:333` for CSV conversion):
```python
{
    'Requirement (Clause)': str,        # Actual requirement text
    'Standard/ Regulation': str,         # Source document
    'Clause': str,                       # Clause number (e.g., "1.7.4.1" or "7.6(a)")
    'Must be included with product?': str,  # Y / N / Ambiguous
    'Requirement Scope': str,            # battery, charger, ebike, bicycle
    'Formatting Requirement(s)?': str,   # Specific formatting rules
    'Required in Print?': str,           # y / n
    'Comments': str                      # Classification notes
}
```

### Consolidation Strategies
Three implementations with different approaches:
- **`consolidate.py`:** Placeholder (Phase 3 not implemented)
- **`consolidate_ai.py`:** Fuzzy pre-filtering + Claude analysis per group
- **`consolidate_smart_ai.py`:** Regulatory intent-based with automatic batching (150 req/batch), uses `ConsolidationGroup` dataclass

Smart AI consolidation respects `min_group_size=3`, `max_group_size=12` and includes timeout handling (600s).

## Code Patterns & Conventions

### Heading Detection Pattern
`extract.py:_is_likely_heading()` uses regex patterns:
```python
r'^\d+(\.\d+)+\s+',         # 1.7.4.1 General
r'^[A-Z]\.\d+\s+',           # A.2.3 Contents
r'^[IVX]+\.\d+\s+',          # VII.4 Instructions
```

### Streamlit State Management
Session state keys track consolidation workflow (`app.py:143-154`):
```python
st.session_state.accepted_groups    # set of group IDs
st.session_state.rejected_groups    # set of group IDs
st.session_state.edited_groups      # dict: group_id → edited text
st.session_state.removed_requirements  # dict: group_id → set of indices
```

### PDF Extraction with Fallback
`extract.py:extract_from_pdf()` tries pdfplumber first, falls back to pypdf on failure:
```python
try:
    pdf = pdfplumber.open(io.BytesIO(file_bytes))
    # ...extract with pdfplumber
except:
    pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
    # ...extract with pypdf
```

## File Organization

**Processing modules** (single responsibility):
- `extract.py` - Text extraction from PDFs, block parsing
- `detect.py` - Pattern matching for sections/clauses (718 lines of regex patterns)
- `classify.py` - Populate 8-column schema with scope detection
- `extract_ai.py` - Claude integration for intelligent extraction

**Consolidation variants** (different approaches, not versions):
- `consolidate.py` - Stub implementation
- `consolidate_ai.py` - Fuzzy + AI hybrid
- `consolidate_smart_ai.py` - Intent-based batching (current best practice)

**Legacy/experimental files** (don't modify unless instructed):
- `app_consolidation_tab_new.py` - Experimental UI iteration
- `consolidate_improved.py` - Older consolidation approach
- `render_consolidation_tab_FIXED.py` - UI component experiment

## Domain-Specific Context

**E-bike standards** in `docs/` include UL 2271, EN 15194, IEC 62133-2, etc.
- Focus on instruction manual/documentation requirements
- Warning/caution/danger labels always required in print
- Scope varies: battery, charger, ebike, bicycle, or combinations
- Clause exclusion patterns filter out false positives (e.g., "manual transmission" vs "instruction manual")

## Testing & Debugging

**No formal test suite** - validate by:
1. Upload test PDF from `docs/` via Streamlit
2. Check extraction results for expected columns
3. Verify consolidation groups make semantic sense
4. Export CSV and spot-check formatting

**Common issues:**
- Anthropic API timeout → Check proxy settings, reduce batch size
- Missing requirements → Adjust patterns in `detect.py`
- Empty RESULTS_STORE → FastAPI server not running or wrong port
