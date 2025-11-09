# Archive

This folder contains deprecated code modules that have been replaced by newer implementations.

## consolidate_improved.py

**Replaced by:** `consolidate_smart_ai.py`

**Reason:** The improved consolidator used rapidfuzz for similarity matching, which was replaced by Claude API-based semantic analysis for better understanding of regulatory intent and meaning.

**Date Archived:** 2025-11-09

**Note:** The `/consolidate_improved` FastAPI endpoint that used this module was also removed as it was not being called by the frontend application.
