"""
Data models for the harmonization layer.

This module defines the core data structures used throughout the harmonization pipeline:
- Clause: Represents a single requirement from a standard
- RequirementGroup: Represents a consolidated group of similar requirements across standards
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Clause:
    """
    Represents a single requirement/clause extracted from a standard.

    This is the atomic unit that gets grouped and consolidated in the harmonization layer.
    """
    clause_number: str          # e.g., "5.1.101", "BB.1.2.a"
    text: str                   # Full requirement text
    standard_name: str          # e.g., "UL 2271", "SS_EN_50604"

    # Phase 0: Regulatory category (assigned by rule-based classifier)
    category: Optional[str] = None            # e.g., "user_documentation_and_instructions"

    # Optional metadata from validate.py tagging
    parent_section: Optional[str] = None      # e.g., "5.1", "BB.1"
    clause_type: Optional[str] = None         # Requirement/Definition/Test_Methodology/Preamble
    mandate_level: Optional[str] = None       # High/Medium/Informative
    safety_flag: Optional[str] = None         # y/n
    manual_flag: Optional[str] = None         # y/n

    # Additional fields from extraction pipeline
    requirement_scope: Optional[str] = None
    formatting_required: Optional[str] = None
    required_in_print: Optional[str] = None
    contains_image: Optional[str] = None
    safety_notice_type: Optional[str] = None
    comments: Optional[str] = None

    def __post_init__(self):
        """Validate required fields are non-empty."""
        if not self.clause_number or not self.clause_number.strip():
            raise ValueError("clause_number cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("text cannot be empty")
        if not self.standard_name or not self.standard_name.strip():
            raise ValueError("standard_name cannot be empty")

    def to_dict(self) -> dict:
        """Convert Clause to dictionary for JSON serialization."""
        return {
            'clause_number': self.clause_number,
            'text': self.text,
            'standard_name': self.standard_name,
            'category': self.category,
            'parent_section': self.parent_section,
            'clause_type': self.clause_type,
            'mandate_level': self.mandate_level,
            'safety_flag': self.safety_flag,
            'manual_flag': self.manual_flag,
            'requirement_scope': self.requirement_scope,
            'formatting_required': self.formatting_required,
            'required_in_print': self.required_in_print,
            'contains_image': self.contains_image,
            'safety_notice_type': self.safety_notice_type,
            'comments': self.comments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Clause':
        """Create Clause from dictionary."""
        return cls(
            clause_number=data['clause_number'],
            text=data['text'],
            standard_name=data['standard_name'],
            category=data.get('category'),
            parent_section=data.get('parent_section'),
            clause_type=data.get('clause_type'),
            mandate_level=data.get('mandate_level'),
            safety_flag=data.get('safety_flag'),
            manual_flag=data.get('manual_flag'),
            requirement_scope=data.get('requirement_scope'),
            formatting_required=data.get('formatting_required'),
            required_in_print=data.get('required_in_print'),
            contains_image=data.get('contains_image'),
            safety_notice_type=data.get('safety_notice_type'),
            comments=data.get('comments'),
        )


@dataclass
class RequirementGroup:
    """
    Represents a consolidated group of similar requirements from MULTIPLE standards.

    This is the output of the harmonization layer - groups that show how different
    standards address the same regulatory intent/compliance goal.

    IMPORTANT: Groups MUST contain clauses from 2+ different standards.
    Groups with all clauses from the same standard should be rejected.
    """
    group_id: int                           # Unique identifier for this group
    core_requirement: str                   # Consolidated/unified requirement text
    clauses: List[Clause] = field(default_factory=list)  # Individual clauses in this group

    # LLM-generated consolidation metadata
    group_title: Optional[str] = None       # Short title for this group (e.g., "Electric Shock Warnings")
    regulatory_intent: Optional[str] = None # Explanation of shared compliance goal
    consolidated_requirement: Optional[str] = None  # Alternative to core_requirement (richer format)
    differences: List[dict] = field(default_factory=list)  # Differences across standards
    unique_requirements: Optional[str] = None  # Requirements unique to specific standards
    conflicts: Optional[str] = None         # Conflicting requirements across standards

    # Legacy metadata (keep for backwards compatibility)
    analysis_notes: Optional[str] = None    # LLM's explanation of why these were grouped
    category: Optional[str] = None          # e.g., "Assembly Instructions", "Battery Safety"

    def __post_init__(self):
        """Validate group integrity."""
        if not self.core_requirement or not self.core_requirement.strip():
            raise ValueError("core_requirement cannot be empty")

        if len(self.clauses) < 2:
            raise ValueError(f"Group must contain at least 2 clauses, got {len(self.clauses)}")

        # CRITICAL: Verify clauses come from multiple standards
        unique_standards = set(clause.standard_name for clause in self.clauses)
        if len(unique_standards) < 2:
            raise ValueError(
                f"Group must contain clauses from 2+ standards, got only: {unique_standards}"
            )

    def get_standards(self) -> List[str]:
        """Return list of unique standard names in this group."""
        return sorted(set(clause.standard_name for clause in self.clauses))

    def get_clause_count(self) -> int:
        """Return number of clauses in this group."""
        return len(self.clauses)

    def to_dict(self) -> dict:
        """Convert RequirementGroup to dictionary for JSON serialization."""
        return {
            'group_id': self.group_id,
            'core_requirement': self.core_requirement,
            'clauses': [clause.to_dict() for clause in self.clauses],
            'group_title': self.group_title,
            'regulatory_intent': self.regulatory_intent,
            'consolidated_requirement': self.consolidated_requirement,
            'differences': self.differences,
            'unique_requirements': self.unique_requirements,
            'conflicts': self.conflicts,
            'analysis_notes': self.analysis_notes,
            'category': self.category,
            'standards': self.get_standards(),
            'clause_count': self.get_clause_count(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RequirementGroup':
        """Create RequirementGroup from dictionary."""
        return cls(
            group_id=data['group_id'],
            core_requirement=data['core_requirement'],
            clauses=[Clause.from_dict(c) for c in data.get('clauses', [])],
            analysis_notes=data.get('analysis_notes'),
            category=data.get('category'),
        )
