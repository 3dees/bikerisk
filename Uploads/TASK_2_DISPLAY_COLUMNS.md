# TASK 2: Update app.py - Display Columns

## What This Does
Adds 2 new columns to the results table: "Contains Image?" and "Safety Notice Type"

## Instructions
1. Open `app.py`
2. Find the `display_results()` function (around line 850)
3. Look for `display_columns = [`
4. Replace that list with the new one below
5. Save

## Find This Code

```python
display_columns = [
    'Description',
    'Standard/Reg',
    'Clause/Requirement',
    'Requirement scope',
    'Formatting required?',
    'Required in Print?',
    'Comments'
]
```

## Replace With This

```python
display_columns = [
    'Description',
    'Standard/Reg',
    'Clause/Requirement',
    'Requirement scope',
    'Formatting required?',
    'Required in Print?',
    'Comments',
    'Contains Image?',      # NEW - flags figure references
    'Safety Notice Type'    # NEW - marks WARNING/DANGER/CAUTION
]
```

## That's It!

Just those 2 lines added to the list.

## ✅ Done?
- [ ] Found the display_columns list
- [ ] Added the 2 new lines
- [ ] Saved the file

**Next:** TASK 3 (adding backward compatibility)
