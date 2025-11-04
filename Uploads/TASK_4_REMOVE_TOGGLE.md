# TASK 4: Update app.py - Remove Rule-Based Toggle

## What This Does
Removes the extraction mode toggle from UI (we're AI-only now).

## Instructions
1. Still in `app.py`
2. Find `render_extraction_tab()` function (around line 140)
3. Find the extraction mode radio buttons
4. Replace that whole section with 2 simple lines
5. Save

## Find This Big Block (Around Line 160)

```python
# Extraction mode toggle
extraction_mode = st.radio(
    "Extraction Mode",
    options=["🤖 AI (Recommended)", "⚙️ Rule-Based (Legacy)"],
    index=0,
    help="AI mode uses Claude for intelligent extraction. Rule-based uses pattern matching."
)

# Convert UI label to API value
mode_value = "ai" if "AI" in extraction_mode else "rules"

# Only show custom section name for rule-based mode
custom_section_name = None
if mode_value == "rules":
    custom_section_name = st.text_input(
        "Custom Section Name (optional)",
        placeholder="e.g., Instruction for use",
        help="Specific section name to look for in the document. Leave blank to use default patterns."
    )
```

## Replace With Just This

```python
# Always use AI mode
mode_value = "ai"
custom_section_name = None
```

## Also Update process_document Call

Find this line (around line 175):
```python
if process_button and uploaded_file:
    process_document(uploaded_file, standard_name, custom_section_name, mode_value)
```

Replace with:
```python
if process_button and uploaded_file:
    process_document(uploaded_file, standard_name, None, "ai")
```

## ✅ Done?
- [ ] Removed extraction mode toggle
- [ ] Updated process_document call
- [ ] Saved the file

**Next:** TASK 5 (updating main.py)
