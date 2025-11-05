# Direct Flow - Quick Reference for Claude Code

## Goal
Add button after PDF extraction that loads data directly into consolidation tab (no download/re-upload).

---

## Changes (4 locations in app.py)

### 1. Add Continue Button (Line ~1095)
**Change:** Replace single "Download CSV" button with 2 buttons side-by-side

**Before:**
```python
st.download_button(
    label="📥 Download CSV",
    ...
)
```

**After:**
```python
col1, col2 = st.columns(2)
with col1:
    st.download_button("📥 Download CSV", ...)
with col2:
    if st.button("➡️ Continue to Consolidation", ...):
        st.session_state.consolidation_df = edited_df.copy()
        # Clear old results
        # Set switch flag
        st.rerun()
```

---

### 2. Handle Tab Switching (Line ~152)
**Change:** Make tab selection dynamic based on session state

**Before:**
```python
tab1, tab2 = st.tabs([...])
with tab1:
    render_extraction_tab()
with tab2:
    render_consolidation_tab()
```

**After:**
```python
# Check if switching to consolidation
if st.session_state.get('switch_to_consolidation'):
    default_tab = 1
    st.session_state.switch_to_consolidation = False
else:
    default_tab = 0

tabs = st.tabs([...])
with tabs[0]:
    render_extraction_tab()
with tabs[1]:
    render_consolidation_tab()
```

---

### 3. Show Extracted Data (Line ~552)
**Change:** Check for extracted data before file upload

**Add this BEFORE file uploader:**
```python
# Check if data from extraction exists
if 'consolidation_df' in st.session_state and uploaded_file is None:
    st.info("📊 Using data from extraction...")
    df = st.session_state.consolidation_df
    st.success(f"✅ Loaded {len(df)} requirements")
    # Show preview
    # Continue to existing code...
```

---

### 4. Add Clear Button (Line ~560)
**Change:** Let user clear extracted data to upload fresh file

**Add after preview:**
```python
if st.button("🗑️ Clear Data (Upload Different File)", ...):
    del st.session_state.consolidation_df
    st.rerun()
```

---

## Test It Works

1. Extract PDF → Click "Continue" → Should switch tabs + load data
2. Go directly to consolidation → Upload CSV → Should work normally
3. Extract PDF → Go to consolidation → Upload different CSV → Should override

---

## Full Code in: DIRECT_FLOW_IMPLEMENTATION.md
## User Guide in: USER_GUIDE_QUICK.md
