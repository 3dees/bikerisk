# Phase 2 Consolidation Prompt Migration

## Summary
Migrated Phase 2 consolidation to use **system + user message structure** with emphasis on **manual-ready, detailed, structured output** and a new **differences_across_standards schema**.

## Changes Made

### 1. Prompt Structure (harmonization/consolidate.py)
- **Old:** Single user message with inline persona
- **New:** Separate system + user messages
  - **System:** Role/persona ("You are a senior regulatory compliance engineer...")
  - **User:** Task with cluster metadata (category, requirement_type) + clauses

### 2. Differences Schema Migration
**Old schema:**
```json
"differences_across_standards": [
  {"standard": "UL 2271", "differences": "text"}
]
```

**New schema:**
```json
"differences_across_standards": [
  {
    "standard_id": "UL 2271",
    "clause_labels": ["5.1.101", "5.2.3"],
    "difference_summary": "Stricter temperature limits..."
  }
]
```

### 3. API Call Signature Changes
**Before:**
```python
call_llm(prompt: str) -> str
```

**After:**
```python
call_llm(system_prompt: str, user_prompt: str) -> str
```

### 4. Files Modified

#### Core Logic
- `harmonization/consolidate.py`
  - `build_llm_prompt_for_group()` → Returns `tuple[str, str]` (system, user)
  - `enrich_group_with_llm()` → Accepts `Callable[[str, str], str]`
  - `consolidate_groups()` → Updated signature
  - `stub_call_llm()` → Updated signature

- `harmonization/pipeline_unify.py`
  - `call_llm_for_consolidation()` → Takes system + user prompts, passes `system=` param to API

- `harmonization/models.py`
  - `RequirementGroup.differences` → Deprecated (kept for backwards compat)
  - `RequirementGroup.differences_across_standards` → New primary field
  - `to_dict()` → Uses new schema

#### Presentation Layer
- `harmonization/report_builder.py`
  - `_build_differences_table()` → Renders 3-column table (Standard | Clauses | Differences)
  - `build_group_html()` → Uses `differences_across_standards`

#### Runners & Tests
- `run_phase2_consolidation.py` → Lambda updated
- `test_small_groups.py` → Lambda updated
- `test_consolidation_regression.py` → Lambda updated

### 5. New Prompt Features

#### Metadata Injection
```python
category_label = CATEGORY_TITLE_MAP.get(category, category)  # Human-readable
requirement_type_label = clause.requirement_type              # Enum string
```

#### Structured Output Emphasis
The prompt now explicitly demands:
- **Manual-ready text** (copy-pasteable)
- **Structured format** with a), b), c) enumeration
- **Complete preservation** of all measurements, conditionals, limits
- **No vague summaries** - detailed, actionable requirements

Example from prompt:
```
Prefer detailed, actionable text like:
"Instructions shall address safe transport, handling, and storage, including:
  a) Stability conditions during use...
  b) Mass information for components...
  c) IF moving could cause damage, describe procedures with warning..."
```

### 6. Backwards Compatibility

**Legacy `differences` field:**
- Still present in `RequirementGroup` model
- Marked as deprecated
- Not populated by new code
- Can map old → new when reading historical files if needed

**Migration strategy:**
- New code writes only `differences_across_standards`
- Old exports can be read with graceful fallback
- Breaking change acceptable per requirements

## Testing

Compile checks passed:
```powershell
python -m py_compile harmonization/consolidate.py
python -m py_compile harmonization/pipeline_unify.py
python -m py_compile harmonization/models.py
```

## Next Steps

1. Run Phase 2 on multi-standard dataset to validate prompt quality
2. Verify HTML report renders new 3-column differences table correctly
3. Test JSON export includes all new schema fields
4. Optional: Create migration script for old JSON files if needed

## API Call Example

```python
from harmonization.pipeline_unify import call_llm_for_consolidation

system_prompt = "You are a senior regulatory compliance engineer..."
user_prompt = "Cluster metadata:\n- Category: User Documentation\n..."

response_json = call_llm_for_consolidation(system_prompt, user_prompt)
# Returns: {"group_title": "...", "differences_across_standards": [...], ...}
```

## Breaking Changes

✅ **Accepted breaking changes:**
- `differences` field deprecated in favor of `differences_across_standards`
- LLM caller signature changed from `(str)` to `(str, str)`
- HTML report table structure changed (2 columns → 3 columns)
- JSON exports use new schema

⚠️ **Impact:**
- Old JSON files won't have `differences_across_standards` populated
- Test files using old lambda signature will fail (fixed in this PR)
- Custom integrations calling `call_llm_for_consolidation` need update
