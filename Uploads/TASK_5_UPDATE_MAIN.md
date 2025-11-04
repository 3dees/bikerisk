# TASK 5: Update main.py - Remove Rule-Based Branch

## What This Does
Removes the rule-based extraction code (we only use AI now).

## Instructions
1. Open `main.py`
2. Find the `upload_file()` function
3. Find the `if extraction_mode == "ai":` block
4. Delete the entire `else:` branch
5. Save

## Find This Section (Around Line 100)

Look for:
```python
# Branch based on extraction mode
if extraction_mode == "ai":
    # HYBRID AI MODE code here
    # ... lots of code ...
    
else:
    # RULE-BASED MODE
    # ... delete everything in this else block ...
```

## What to Do

1. **KEEP** everything in the `if extraction_mode == "ai":` block
2. **DELETE** the entire `else:` block (all the rule-based code)

The result should just be:
```python
# Always use HYBRID AI MODE
if extraction_mode == "ai":
    # ... keep all this AI code ...
    
# No else block anymore
```

## If You're Unsure

The `else:` block starts around line 150 and ends around line 180.

Delete from:
```python
else:
    # RULE-BASED MODE
```

To just before:
```python
    # Store results
    RESULTS_STORE[job_id] = {
```

## Alternative: Just Comment It Out

If you're nervous about deleting:
```python
# else:
#     # RULE-BASED MODE (deprecated)
#     # ... commented out code ...
```

## ✅ Done?
- [ ] Found the if/else block
- [ ] Deleted (or commented) the else branch
- [ ] Saved the file

**Next:** TASK 6 (testing!)
