# BikeRisk - E-Bike Standards Requirement Extractor

## Development Environment

### Python Version
- **Required:** Python 3.10+
- **Tested on:** Python 3.12.3
- Use `python3` or `python` depending on your system setup

### Dependency Management
- All dependencies are listed in `requirements.txt`
- Install with: `pip install -r requirements.txt`
- Key dependencies:
  - FastAPI 0.109.0 (backend API)
  - Streamlit 1.31.0 (frontend UI)
  - Anthropic >=0.72.0 (AI integration)
  - pdfplumber 0.11.0 + pypdf 4.0.1 (PDF extraction)
  - rapidfuzz 3.6.1 (text similarity)
  - python-dotenv 1.0.0 (environment variables)

### Environment Variables
- Create a `.env` file in the project root (never commit this file)
- Required variables:
  - `ANTHROPIC_API_KEY=sk-ant-...` (for AI extraction modes)
- The `.env` file is excluded in `.gitignore`

## Code Style & Conventions

### Python Style
- Follow PEP 8 conventions
- Use descriptive variable names (e.g., `manual_sections`, `detected_clauses`)
- No formal linter configured - maintain consistency with existing code
- Maximum line length: flexible, but keep readability in mind

### Naming Conventions
- **Files:** lowercase with underscores (e.g., `extract_ai.py`, `consolidate_smart_ai.py`)
- **Functions:** lowercase with underscores (e.g., `extract_from_pdf`, `detect_manual_sections`)
- **Classes:** PascalCase (e.g., `ConsolidationGroup`)
- **Constants:** UPPERCASE with underscores (e.g., `RESULTS_STORE`, `MANUAL_SECTION_PATTERNS`)

### Comments & Documentation
- Add comments for complex regex patterns and business logic
- Document function parameters and return values for public APIs
- Keep domain-specific explanations (e.g., e-bike standard clause formats)

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

## Security Best Practices

### API Key Handling
- **NEVER** commit API keys to version control
- Always use `.env` file for secrets (excluded in `.gitignore`)
- The Anthropic API key pattern is: `ANTHROPIC_API_KEY=sk-ant-...`
- Streamlit allows session-level override in sidebar for testing
- Use environment variables: `os.getenv('ANTHROPIC_API_KEY')`

### Proxy Configuration
- NO_PROXY pattern for Anthropic API: `os.environ['NO_PROXY'] = '*.anthropic.com'`
- This is required for some network environments (see `extract_ai.py:47`)

### File Upload Security
- PDF files are processed in-memory (no disk writes by default)
- Uploaded files go to `Uploads/` directory (excluded from git)
- Validate file types before processing
- Be cautious with OCR features (not yet implemented)

## Error Handling Patterns

### PDF Extraction Fallback
Always use the two-tier fallback pattern:
```python
try:
    # Try pdfplumber first (high quality)
    pdf = pdfplumber.open(io.BytesIO(file_bytes))
    # ... process
except Exception as e:
    # Fall back to pypdf
    pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
    # ... process
```

### API Error Handling
- FastAPI endpoints should raise `HTTPException` with appropriate status codes
- Include descriptive error messages
- Log errors for debugging but don't expose sensitive data

### AI Extraction Error Handling
- Handle Anthropic API timeouts (default: 600s)
- Implement fallback to rule-based extraction when AI fails
- Return partial results when possible
- Set confidence scores based on extraction method used

## Contributing Guidelines for Copilot

### Making Code Changes
- **Minimal changes:** Only modify files directly related to the issue
- **Don't refactor:** Avoid large-scale refactoring unless explicitly requested
- **Preserve working code:** Don't delete or modify working functionality
- **Test manually:** Run the application and verify changes work as expected

### Adding New Features
- Follow existing patterns (see Code Patterns & Conventions section)
- Update copilot-instructions.md if adding new architectural patterns
- Add new dependencies to `requirements.txt` only if necessary
- Document new environment variables in this file

### Working with Legacy Files
These files are experimental/legacy - **don't modify** unless instructed:
- `app_consolidation_tab_new.py`
- `consolidate_improved.py`
- `render_consolidation_tab_FIXED.py`
- `extract_ai_OLD.py`
- `extract_ai_improved.py`

### Branch and PR Conventions
- Branch naming: Use descriptive names (e.g., `feature/add-ocr`, `fix/pdf-timeout`, `docs/update-readme`)
- PR titles: Clear, concise description of changes
- PR descriptions: Include:
  - What was changed and why
  - How to test the changes
  - Any breaking changes or new dependencies
  - Screenshots for UI changes

### Testing Changes
Since there's no formal test suite:
1. Start both servers: `python main.py` and `streamlit run app.py`
2. Upload a test PDF from `docs/` directory
3. Verify extraction results match expected schema
4. Check consolidation groups make sense
5. Export CSV and verify formatting
6. Test error cases (invalid PDFs, missing API keys)

## Common Pitfalls to Avoid

### Don't Do This:
- ❌ Commit `.env` files or API keys
- ❌ Modify `RESULTS_STORE` structure without updating all consumers
- ❌ Change the 8-column schema without updating `classify.py` and `app.py`
- ❌ Add new regex patterns without testing against real PDFs
- ❌ Remove existing detection patterns (may break specific standards)
- ❌ Modify legacy/experimental files
- ❌ Force-install different versions of core dependencies

### Do This Instead:
- ✅ Use `.env` for all secrets
- ✅ Test with real PDFs from `docs/` directory
- ✅ Add new patterns alongside existing ones
- ✅ Preserve backward compatibility
- ✅ Update documentation when adding features
- ✅ Follow existing code style and patterns
- ✅ Handle errors gracefully with fallbacks
