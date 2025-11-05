# TASK 3: Update app.py - Add Column Defaults

## What This Does
Adds default values for the new columns so old saved projects don't crash.

## Instructions
1. Still in `app.py`
2. Find the `normalize_column_names()` function (around line 270)
3. Scroll to the BOTTOM of that function
4. Right before `return df_normalized`, add 4 new lines
5. Save

## Find the End of This Function

Look for:
```python
def normalize_column_names(df):
    # ... lots of code ...
    
    # Apply mapping
    df_normalized = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df_normalized.columns and new_name not in df_normalized.columns:
            df_normalized.rename(columns={old_name: new_name}, inplace=True)
    
    return df_normalized  # <-- ADD CODE BEFORE THIS LINE
```

## Add These 4 Lines Before the `return`

```python
    # Add defaults for new columns (backward compatibility)
    if 'Contains Image?' not in df_normalized.columns:
        df_normalized['Contains Image?'] = 'N'
    if 'Safety Notice Type' not in df_normalized.columns:
        df_normalized['Safety Notice Type'] = 'None'
    
    return df_normalized
```

## The Full End Should Look Like

```python
    # Apply mapping
    df_normalized = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df_normalized.columns and new_name not in df_normalized.columns:
            df_normalized.rename(columns={old_name: new_name}, inplace=True)
    
    # Add defaults for new columns (backward compatibility)
    if 'Contains Image?' not in df_normalized.columns:
        df_normalized['Contains Image?'] = 'N'
    if 'Safety Notice Type' not in df_normalized.columns:
        df_normalized['Safety Notice Type'] = 'None'
    
    return df_normalized
```

## ✅ Done?
- [ ] Found normalize_column_names function
- [ ] Added 4 lines before the return
- [ ] Saved the file

**Next:** TASK 4 (removing UI toggle)
