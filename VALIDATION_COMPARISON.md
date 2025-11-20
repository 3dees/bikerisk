# Validation Filtering Comparison

## Philosophy Change
**OLD Approach**: Aggressive filtering - "Remove anything that looks like garbage"
**NEW Approach**: Minimal filtering - "When in doubt, include it"

## Rationale
Better to over-include and let manual review prune false positives than to miss requirements during automated filtering.

## SS_EN_50604 Results

### Old Aggressive Filtering
- Input: 178 requirements
- Output: **48 valid** (27%)
- Removed: 130 (73%)
- Top removal reasons:
  - 41 missing required keywords
  - 32 Section 3 definitions
  - 28 informative test annexes (FF/GG)
  - 15 too short
  - 4 test procedures

### New Minimal Filtering
- Input: 178 requirements
- Output: **144 valid** (81%)
- Removed: 34 (19%)
- Top removal reasons:
  - 25 pure definitions (Section 3, no requirement keywords)
  - 4 too short (< 10 chars)
  - 3 pure test methodology
  - 2 too short (9 chars)

## Key Differences

### What's Now KEPT (was previously removed):
1. **Scope sections** - May contain applicability requirements
2. **Normative references** - May specify compliance requirements
3. **All annexes** (even informative) - May contain requirements
4. **Test requirements** - Anything with "shall/must/required" language
5. **Marking/instruction items** - All items with requirement keywords
6. **Informative content** - If it has requirement language, keep it

### What's Still REMOVED:
1. **Pure definitions** - Section 3 items with NO requirement keywords
2. **N/A placeholders** - Empty structural headings
3. **Preamble** - "This clause of ISO X is applicable" (no additional content)
4. **Pure test methodology** - No requirement language (just procedure description)
5. **Extremely short** - < 10 characters

## New Features

### Parent Section Recognition
Automatically populates "Parent Section" field by parsing clause numbers:
- "5.1.101" → "5. General requirements"
- "BB.1.1" → "Annex BB - Marking and instructions"
- "7.2.101.a" → "7. Environmental requirements"

### Comments Field Cleanup
Strips extraction metadata ("GPT extraction", "N/A") from Comments field during validation.

## Impact
**Old**: 178 → 48 (missed 96 valid requirements)
**New**: 178 → 144 (more aligned with manual baseline of 95-135 valid reqs)

The new approach captures 3x more requirements while only removing obvious non-requirements.
