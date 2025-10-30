# E-Bike Standards Requirement Extractor

Extract instruction/manual requirements from e-bike standards and regulations (PDF files).

**Current Version:** Phase 1-2 (Extraction + Classification)

## Features

### Phase 1: Extraction Pipeline ✅
- ✅ PDF text extraction with fallback (pdfplumber → pypdf)
- ✅ Structured text block parsing with heading detection
- ✅ Extraction quality confidence scoring
- ✅ Error handling with partial results

### Phase 2: Detection & Classification ✅
- ✅ **Pass A:** Detect dedicated manual/instructions sections
- ✅ **Pass B:** Find scattered manual-related clauses throughout document
- ✅ Rule-based classification into 8-column schema
- ✅ Confidence scoring per requirement
- ✅ Automatic scope detection (battery, charger, ebike, bicycle)
- ✅ Formatting requirement extraction
- ✅ Print requirement detection

### Phase 3: Consolidation (Coming Soon)
- ⏳ Text normalization and similarity matching
- ⏳ Group similar requirements with explanations
- ⏳ Preserve numerical and scope differences

---

## Output Schema

Each extracted requirement includes these 8 fields:

| Field | Description | Example |
|-------|-------------|---------|
| **Requirement (Clause)** | Actual requirement text | "A bicycle shall have an instruction manual..." |
| **Standard/ Regulation** | Source document | "16 CFR Part 1512" |
| **Clause** | Clause number/reference | "19.1.3" or "1.7.4.1 (a)" |
| **Must be included with product?** | Y / N / Ambiguous | "Y" |
| **Requirement Scope** | battery, charger, ebike, bicycle | "ebike, bicycle" |
| **Formatting Requirement(s)?** | Specific formatting rules | "shall be in capital letters" |
| **Required in Print?** | y / n | "y" |
| **Comments** | Classification notes | "inside instructions section" |

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

1. **Clone or navigate to the project directory**
   ```bash
   cd /home/user/bikerisk
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Start the Backend (FastAPI)

In one terminal:
```bash
python main.py
```

The API will run on `http://localhost:8000`

API Endpoints:
- `POST /upload` - Upload PDF for processing
- `GET /results/{job_id}` - Get extraction results
- `GET /export/{job_id}` - Export results as CSV
- `GET /jobs` - List all processed jobs

### Start the Frontend (Streamlit)

In another terminal:
```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

### Using the Web UI

1. **Upload a PDF** containing an e-bike standard or regulation
2. **Optionally specify** the standard name (e.g., "EN 15194")
3. **Click "Process Document"** to extract requirements
4. **View results** in the interactive table
5. **Filter results** by inclusion status, print requirement, or scope
6. **Download CSV** for further analysis

---

## Project Structure

```
bikerisk/
├── main.py           # FastAPI backend
├── app.py            # Streamlit frontend
├── extract.py        # PDF text extraction with fallback
├── detect.py         # Manual section & clause detection (Pass A + B)
├── classify.py       # Classification rules & schema
├── consolidate.py    # Consolidation logic (Phase 3 placeholder)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## How It Works

### 1. Extraction (extract.py)
- Attempts extraction with `pdfplumber` first (high confidence)
- Falls back to `pypdf` if needed (medium confidence)
- Returns structured text blocks with line numbers and heading detection

### 2. Detection (detect.py)

**Pass A: Section-Based Detection**
- Finds sections with headings containing:
  - "instructions for use"
  - "information for use"
  - "content of the instructions"
  - "requirements for manuals"
  - "marking and instructions"
  - "user information"
  - "accompanying documents"
  - Patterns like `1.7.4` or `7.3.2`

**Pass B: Clause-Based Detection**
- Scans entire document for clauses containing:
  - "shall accompany"
  - "shall be supplied with"
  - "shall be included with the product"
  - "shall be provided to the user"
  - "WARNING", "CAUTION", "DANGER"
  - "instruction manual"
  - "user instructions"

### 3. Classification (classify.py)

Applies rule-based logic:

**Must be included with product?**
- **Y** if: inside instructions section, has hard requirement phrase, or contains warning tokens
- **Ambiguous** if: has soft requirement phrase ("shall be made available")
- **N** if: keyword match only

**Required in Print?**
- **y** if: explicit print mention, warning token, or from instructions section
- **n** otherwise

**Requirement Scope**
- Auto-detected from keywords: battery, charger, ebike, bicycle

**Formatting Requirements**
- Extracted from phrases like:
  - "shall be in capital letters"
  - "shall be clearly legible"
  - "shall bear the words 'Original instructions'"

---

## Detection Patterns

### Clause Number Patterns
- `1.7.4.1` - Standard hierarchical numbering
- `A.2.3` - Annex-style numbering
- `VII.4` - Roman numeral sections
- `1.2(a)` - Parenthetical sub-clauses

### Warning Tokens
- `WARNING`, `CAUTION`, `DANGER`, `NOTE`
- Automatically marked as required in print

---

## Future Enhancements

### Ready to Add (marked in code)
- **Phase 3:** Consolidation with similarity matching (rapidfuzz)
- **Additional formats:** .doc, .docx, .xls, .xlsx, HTML via URL
- **OCR fallback:** pytesseract for image-based PDFs
- **Toggle controls:** User-adjustable "Required in Print" in UI
- **Rules admin:** Configure detection patterns and thresholds
- **Cloud storage:** Replace in-memory storage with database
- **Multi-language:** Extract and preserve language indicators
- **Version tracking:** Diff between standard versions

### Marked for Extension
- See comments starting with `# TODO:` in the code
- See comments starting with `# Future:` in the code

---

## Testing

### Test with Sample Standards

Recommended public standards to test:
- **EN 15194** - Electrically power assisted cycles (EPAC)
- **EN 50604-1** - Battery requirements for e-bikes
- **16 CFR Part 1512** - Requirements for bicycles (US)
- **Machinery Directive 2006/42/EC** - Section 1.7.4 on instructions

### Manual Testing Checklist
1. Upload a PDF
2. Verify extraction method and confidence shown
3. Check that manual sections are detected (Pass A)
4. Check that scattered clauses are detected (Pass B)
5. Verify classification rules applied correctly
6. Test filters in UI
7. Download CSV and verify format

---

## Troubleshooting

### "Cannot connect to backend API"
- Ensure FastAPI is running: `python main.py`
- Check that port 8000 is not in use

### "Extraction produced insufficient content"
- PDF may be image-based (needs OCR - not yet implemented)
- Try converting PDF to text format first
- Check if PDF is corrupted

### No Requirements Found
- Document may not contain manual/instructions sections
- Check "Extraction Information" to see what was extracted
- Patterns may need adjustment for this specific standard

### Poor Detection Quality
- Some standards use non-standard heading formats
- May need to add custom patterns in `detect.py`
- Check `_confidence` field in results for quality indicators

---

## Contributing

This is an internal tool for e-bike standards analysis. To extend:

1. **Add new patterns:** Edit `detect.py` or `classify.py`
2. **Add new file formats:** Extend `extract.py`
3. **Improve UI:** Modify `app.py`
4. **Add consolidation:** Implement `consolidate.py` (Phase 3)

---

## License

Internal use only.

---

## Contact

For questions or issues, contact the project maintainer.
