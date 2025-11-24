"""
Test script for Phase 0.5 requirement type classification.

This validates that the classify_requirement_type function correctly
identifies different requirement types based on text content.
"""

from harmonization.models import Clause
from harmonization.grouping import (
    classify_requirement_type,
    REQUIREMENT_TYPE_TEST_METHOD,
    REQUIREMENT_TYPE_TEST_ACCEPTANCE,
    REQUIREMENT_TYPE_LABELING,
    REQUIREMENT_TYPE_WARNINGS,
    REQUIREMENT_TYPE_USER_INSTRUCTIONS,
    REQUIREMENT_TYPE_CHARGING_STORAGE,
    REQUIREMENT_TYPE_DEFINITIONS,
    REQUIREMENT_TYPE_BMS_BEHAVIOR,
    REQUIREMENT_TYPE_DESIGN_CONSTRUCTION,
)


def test_requirement_type_classification():
    """Test requirement type classification with sample clauses."""
    
    test_cases = [
        # (clause_text, clause_number, expected_type, description)
        (
            "The following test shall be carried out to verify compliance with section 5.1",
            "5.1.2",
            REQUIREMENT_TYPE_TEST_METHOD,
            "Test method clause"
        ),
        (
            "No fire shall occur during the test. The battery shall not rupture or explode.",
            "5.1.3",
            REQUIREMENT_TYPE_TEST_ACCEPTANCE,
            "Test acceptance criteria"
        ),
        (
            "The battery shall be marked with the manufacturer's name and rated voltage.",
            "7.2.1",
            REQUIREMENT_TYPE_LABELING,
            "Labeling requirement"
        ),
        (
            "Warning: Do not disassemble. Risk of electric shock.",
            "7.3.1",
            REQUIREMENT_TYPE_WARNINGS,
            "Warning text"
        ),
        (
            "Instructions for use shall include charging temperature limits and storage conditions.",
            "8.1",
            REQUIREMENT_TYPE_CHARGING_STORAGE,
            "Charging/storage instructions"
        ),
        (
            "The instructions shall include information about proper disposal and recycling.",
            "8.2",
            REQUIREMENT_TYPE_USER_INSTRUCTIONS,
            "User instructions (non-charging)"
        ),
        (
            "For the purposes of this standard, the following definitions apply.",
            "3.1",
            REQUIREMENT_TYPE_DEFINITIONS,
            "Definitions section"
        ),
        (
            "The battery management system shall provide overcharge protection.",
            "4.3.1",
            REQUIREMENT_TYPE_BMS_BEHAVIOR,
            "BMS requirement"
        ),
        (
            "The enclosure shall be constructed to provide mechanical strength.",
            "4.2.1",
            REQUIREMENT_TYPE_DESIGN_CONSTRUCTION,
            "Design/construction requirement"
        ),
    ]
    
    print("=" * 80)
    print("PHASE 0.5 REQUIREMENT TYPE CLASSIFICATION TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for text, clause_num, expected_type, description in test_cases:
        clause = Clause(
            clause_number=clause_num,
            text=text,
            standard_name="TEST_STANDARD"
        )
        
        actual_type = classify_requirement_type(clause)
        
        status = "✓ PASS" if actual_type == expected_type else "✗ FAIL"
        if actual_type == expected_type:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} - {description}")
        print(f"  Clause: {clause_num}")
        print(f"  Text: {text[:60]}...")
        print(f"  Expected: {expected_type}")
        print(f"  Actual:   {actual_type}")
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = test_requirement_type_classification()
    exit(0 if success else 1)
