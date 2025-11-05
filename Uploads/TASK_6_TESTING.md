# TASK 6: Testing

## What This Does
Verifies everything works after your changes.

## Before You Start

**Restart both servers:**

Terminal 1:
```bash
# Stop if running (Ctrl+C)
python main.py
```

Terminal 2:
```bash
# Stop if running (Ctrl+C)
streamlit run app.py
```

Wait for both to fully start before testing.

---

## Test 1: Basic Upload ✅

**Steps:**
1. Open app in browser
2. Go to "Extract from PDFs" tab
3. Upload ANY PDF standard
4. Click "Process Document"

**Check:**
- [ ] No errors
- [ ] Results table appears
- [ ] Table has 9 columns (not 7)
- [ ] See "Contains Image?" column
- [ ] See "Safety Notice Type" column

**If it fails:** Check terminal for errors

---

## Test 2: New Columns Have Data ✅

**Steps:**
1. Look at the results table
2. Scroll right to see all columns

**Check:**
- [ ] "Contains Image?" shows "N" or "Y - Figure X"
- [ ] "Safety Notice Type" shows "None" or "WARNING" etc.

**If empty:** That's OK if your PDF has no images/warnings

---

## Test 3: Export Works ✅

**Steps:**
1. Click "Download CSV" button
2. Open the CSV in Excel/Google Sheets

**Check:**
- [ ] CSV has 9 columns
- [ ] New columns are present
- [ ] Data looks correct

---

## Test 4: Old Projects Don't Crash ✅

**Only if you have saved projects:**

**Steps:**
1. Go to consolidation tab
2. Try to load an old project

**Check:**
- [ ] Project loads without error
- [ ] If old data shows, new columns default to "N" and "None"

**If it crashes:** Check Task 3 was done correctly

---

## Test 5: Consolidation Still Works ✅

**Steps:**
1. Upload a spreadsheet to consolidation tab
2. Click "Analyze with Smart AI"

**Check:**
- [ ] Consolidation runs normally
- [ ] No column errors
- [ ] Results display correctly

---

## Test 6: Image Detection ✅

**Only if you have a PDF with figures:**

**Steps:**
1. Upload a PDF that mentions "Figure X" or "Table X"
2. Process it

**Check:**
- [ ] "Contains Image?" shows "Y - Figure X.X"

**If not detected:** That's a nice-to-have, not critical

---

## Test 7: Safety Notices ✅

**Only if you have a PDF with WARNING/DANGER:**

**Steps:**
1. Upload a PDF with safety warnings
2. Process it

**Check:**
- [ ] "Safety Notice Type" shows "WARNING" or "DANGER"
- [ ] Safety notices are extracted

**If not detected:** That's a nice-to-have, not critical

---

## 🎉 All Tests Pass?

If Tests 1-5 pass, you're good!

Tests 6-7 are bonus features.

---

## 🚨 If Something Failed

### Error: "Column not found"
→ Go back to TASK 3, make sure you added the defaults

### Error: "Module not found"
→ Make sure `extract_ai.py` is in project root

### No extraction results
→ Check API key in .env file

### Old rule-based toggle still showing
→ Go back to TASK 4, remove the UI toggle

### Results table only has 7 columns
→ Go back to TASK 2, add the 2 new columns

---

## ✅ Success Criteria

**Minimum to be working:**
- [x] App starts without errors
- [x] Can upload and process PDFs
- [x] Results show 9 columns
- [x] Can export CSV
- [x] No crashes

**Bonus (nice to have):**
- [ ] Images detected
- [ ] Safety notices flagged
- [ ] Old projects still load

---

## You're Done! 🎉

If Tests 1-5 passed, everything critical is working.

**What changed:**
- ✅ Better extraction accuracy
- ✅ Image flagging
- ✅ Safety notice marking
- ✅ 2 new columns in results
- ✅ AI-only mode

**Next steps:**
- Use it with real PDFs
- See if extraction quality improved
- Report back any issues
