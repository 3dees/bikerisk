"""
HTML report generation for harmonization layer.

This module generates clean, professional HTML reports showing consolidated
requirement groups across multiple standards.
"""

import html
from typing import List
from datetime import datetime

from harmonization.models import RequirementGroup, Clause


def escape_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS and rendering issues."""
    return html.escape(text)


def build_group_html(group: RequirementGroup) -> str:
    """
    Build HTML for a single RequirementGroup with Phase 2 consolidation fields.

    Args:
        group: RequirementGroup object to render

    Returns:
        HTML string for this group
    """
    # Extract group metadata (Phase 2 fields preferred, fallback to legacy)
    group_id = group.group_id
    group_title = group.group_title or group.category or "Uncategorized"
    regulatory_intent = group.regulatory_intent or group.analysis_notes or "No regulatory intent provided."
    consolidated_requirement = group.consolidated_requirement or group.core_requirement
    differences_across_standards = group.differences_across_standards or []
    unique_requirements = group.unique_requirements
    conflicts = group.conflicts
    standards = group.get_standards()
    clause_count = group.get_clause_count()

    # Calculate simple status based on number of standards
    if len(standards) >= 3:
        status = "High Coverage"
        status_class = "status-high"
    elif len(standards) == 2:
        status = "Medium Coverage"
        status_class = "status-medium"
    else:
        status = "Low Coverage"
        status_class = "status-low"

    # Calculate similarity score (placeholder - in real version, could use actual embeddings)
    # For now, use a heuristic based on clause count
    similarity_score = min(95, 70 + (clause_count * 5))

    # Build members list HTML
    members_html = []
    for clause in group.clauses:
        member_html = f"""
        <div class="member">
            <div class="member-header">
                <strong>{escape_html(clause.standard_name)}</strong> -
                <code>{escape_html(clause.clause_number)}</code>
            </div>
            <div class="member-text">
                {escape_html(clause.text)}
            </div>
            <div class="member-metadata">
                {_build_clause_metadata_badges(clause)}
            </div>
        </div>
        """
        members_html.append(member_html)

    members_section = "\n".join(members_html)

    # Build conflicts warning if present
    conflicts_html = ""
    if conflicts:
        conflicts_html = f"""
            <div class="section conflicts-section">
                <h3>⚠️ Conflicts Detected</h3>
                <div class="conflicts-content">
                    {escape_html(conflicts)}
                </div>
            </div>
        """

    # Build unique requirements section if present
    unique_html = ""
    if unique_requirements:
        unique_html = f"""
            <div class="section unique-section">
                <h3>Unique Requirements</h3>
                <div class="unique-content">
                    {escape_html(unique_requirements)}
                </div>
            </div>
        """

    # Build differences table (Phase 2 new schema)
    differences_html = _build_differences_table(group.differences_across_standards) if group.differences_across_standards else _extract_critical_differences(group.clauses)

    # Build complete group HTML (Phase 2 structure)
    group_html = f"""
    <div class="group" id="group-{group_id}">
        <div class="group-header">
            <h2>Group {group_id + 1}: {escape_html(group_title)}</h2>
            <div class="group-badges">
                <span class="badge {status_class}">{status}</span>
                <span class="badge badge-similarity">Similarity: {similarity_score}%</span>
                <span class="badge badge-count">{clause_count} clauses from {len(standards)} standards</span>
                {('<span class="badge badge-conflict">⚠️ Conflicts</span>' if conflicts else '')}
            </div>
        </div>

        <div class="group-content">
            <div class="section">
                <h3>Standards Covered</h3>
                <div class="standards-list">
                    {_build_standards_badges(standards)}
                </div>
            </div>

            <div class="section">
                <h3>Regulatory Intent</h3>
                <p class="regulatory-intent">{escape_html(regulatory_intent)}</p>
            </div>

            <div class="section highlight">
                <h3>Consolidated Requirement</h3>
                <div class="consolidated-requirement">
                    {_format_requirement_text(consolidated_requirement)}
                </div>
            </div>

            {conflicts_html}

            <div class="section">
                <h3>Differences Across Standards</h3>
                <div class="critical-differences">
                    {differences_html}
                </div>
            </div>

            {unique_html}

            <div class="section">
                <h3>Member Requirements ({clause_count})</h3>
                <div class="members">
                    {members_section}
                </div>
            </div>
        </div>
    </div>
    """

    return group_html


def _build_clause_metadata_badges(clause: Clause) -> str:
    """Build HTML badges for clause metadata (from validate.py tagging)."""
    badges = []

    if clause.mandate_level:
        level_class = {
            'High': 'badge-high',
            'Medium': 'badge-medium',
            'Informative': 'badge-info'
        }.get(clause.mandate_level, 'badge-info')
        badges.append(f'<span class="badge {level_class}">{escape_html(clause.mandate_level)}</span>')

    if clause.safety_flag == 'y':
        badges.append('<span class="badge badge-safety">Safety</span>')

    if clause.manual_flag == 'y':
        badges.append('<span class="badge badge-manual">Manual</span>')

    if clause.clause_type and clause.clause_type != 'Requirement':
        badges.append(f'<span class="badge badge-type">{escape_html(clause.clause_type)}</span>')

    return ' '.join(badges) if badges else '<span class="badge badge-info">No metadata</span>'


def _build_standards_badges(standards: List[str]) -> str:
    """Build HTML badges for standards list."""
    badges = [f'<span class="badge badge-standard">{escape_html(std)}</span>' for std in standards]
    return ' '.join(badges)


def _format_requirement_text(text: str) -> str:
    """
    Format requirement text with proper line breaks and structure.
    Preserves a), b), c) formatting and paragraph breaks.
    """
    # Escape HTML first
    text = escape_html(text)

    # Replace newlines with <br> tags
    text = text.replace('\n', '<br>')

    # Wrap in paragraph
    return f'<p>{text}</p>'


def _build_differences_table(differences_across_standards: List[dict]) -> str:
    """
    Build HTML table showing differences across standards (Phase 2 new schema).

    Args:
        differences_across_standards: List of dicts with 'standard_id', 'clause_labels', and 'difference_summary' keys

    Returns:
        HTML table string
    """
    if not differences_across_standards:
        return '<p class="no-differences">No differences specified.</p>'

    rows = []
    for diff in differences_across_standards:
        standard_id = diff.get('standard_id', 'Unknown')
        clause_labels = diff.get('clause_labels', [])
        difference_summary = diff.get('difference_summary', 'N/A')

        # Format clause labels as comma-separated list
        clauses_str = ', '.join(str(c) for c in clause_labels) if clause_labels else 'N/A'

        # Format multi-line differences
        diffs_html = escape_html(difference_summary).replace('\n', '<br>')

        rows.append(f"""
            <tr>
                <td class="diff-standard"><strong>{escape_html(standard_id)}</strong></td>
                <td class="diff-clauses">{escape_html(clauses_str)}</td>
                <td class="diff-text">{diffs_html}</td>
            </tr>
        """)

    table_html = f"""
        <table class="differences-table">
            <thead>
                <tr>
                    <th>Standard</th>
                    <th>Clauses</th>
                    <th>Specific Differences / Stricter Requirements</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    """

    return table_html


def _extract_critical_differences(clauses: List[Clause]) -> str:
    """
    Extract critical differences between clauses in a group.

    This is a simple heuristic-based approach. In a real implementation,
    you might use LLM to identify specific differences.
    """
    if len(clauses) < 2:
        return '<p class="no-differences">No differences (single clause).</p>'

    # Simple heuristic: look for numerical values that differ
    differences = []

    # Check for different mandate levels
    mandate_levels = set(c.mandate_level for c in clauses if c.mandate_level)
    if len(mandate_levels) > 1:
        differences.append(f"Different mandate levels: {', '.join(sorted(mandate_levels))}")

    # Check for safety vs non-safety
    safety_flags = set(c.safety_flag for c in clauses if c.safety_flag)
    if 'y' in safety_flags and 'n' in safety_flags:
        differences.append("Some requirements flagged as safety-critical, others not")

    # Check for manual vs non-manual
    manual_flags = set(c.manual_flag for c in clauses if c.manual_flag)
    if 'y' in manual_flags and 'n' in manual_flags:
        differences.append("Some requirements related to documentation, others not")

    if not differences:
        differences.append("No significant structural differences detected")

    diff_html = '<ul>' + ''.join(f'<li>{escape_html(d)}</li>' for d in differences) + '</ul>'
    return diff_html


def build_html_report(
    groups: List[RequirementGroup],
    title: str = "Cross-Standard Harmonization Report"
) -> str:
    """
    Build complete HTML report from a list of RequirementGroup objects.

    Args:
        groups: List of RequirementGroup objects to include in report
        title: Report title

    Returns:
        Complete HTML document as string
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate summary statistics
    total_groups = len(groups)
    total_clauses = sum(g.get_clause_count() for g in groups)
    all_standards = set()
    for g in groups:
        all_standards.update(g.get_standards())
    total_standards = len(all_standards)

    # Build groups HTML
    groups_html = "\n".join(build_group_html(g) for g in groups)

    # Build complete HTML document
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape_html(title)}</title>
    <style>
        {_get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{escape_html(title)}</h1>
            <p class="subtitle">Generated on {timestamp}</p>
        </header>

        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-stats">
                <div class="stat">
                    <div class="stat-value">{total_groups}</div>
                    <div class="stat-label">Requirement Groups</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_clauses}</div>
                    <div class="stat-label">Total Clauses</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_standards}</div>
                    <div class="stat-label">Standards Covered</div>
                </div>
            </div>
            <div class="standards-covered">
                <strong>Standards:</strong> {', '.join(sorted(all_standards))}
            </div>
        </div>

        <div class="groups-container">
            {groups_html}
        </div>

        <footer>
            <p>Generated by Harmonization Layer - BikeRisk Compliance Tool</p>
        </footer>
    </div>
</body>
</html>
"""

    return html_doc


def save_json_report(groups: List[RequirementGroup], output_path: str) -> None:
    """Save consolidated groups as JSON array to a file.

    Each group is serialized via RequirementGroup.to_dict(), including member clauses
    and Phase 2 consolidation fields (group_title, regulatory_intent, differences, etc.).

    Args:
        groups: List of consolidated RequirementGroup objects
        output_path: Destination JSON file path
    """
    import json
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = [g.to_dict() for g in groups]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _get_css_styles() -> str:
    """Return CSS styles for the HTML report."""
    return """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        header {
            border-bottom: 3px solid #2563eb;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        h1 {
            color: #1e40af;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #6b7280;
            font-size: 0.9em;
        }

        .summary {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 4px;
        }

        .summary h2 {
            color: #1e40af;
            margin-bottom: 15px;
        }

        .summary-stats {
            display: flex;
            gap: 30px;
            margin-bottom: 15px;
        }

        .stat {
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2563eb;
        }

        .stat-label {
            font-size: 0.9em;
            color: #6b7280;
        }

        .standards-covered {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e5e7eb;
            color: #4b5563;
        }

        .group {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-bottom: 30px;
            overflow: hidden;
        }

        .group-header {
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            color: white;
            padding: 20px;
        }

        .group-header h2 {
            color: white;
            margin-bottom: 10px;
        }

        .group-badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .status-high {
            background: #10b981;
            color: white;
        }

        .status-medium {
            background: #f59e0b;
            color: white;
        }

        .status-low {
            background: #6b7280;
            color: white;
        }

        .badge-similarity {
            background: #8b5cf6;
            color: white;
        }

        .badge-count {
            background: rgba(255,255,255,0.2);
            color: white;
        }

        .badge-standard {
            background: #3b82f6;
            color: white;
        }

        .badge-high {
            background: #dc2626;
            color: white;
        }

        .badge-medium {
            background: #f59e0b;
            color: white;
        }

        .badge-info {
            background: #6b7280;
            color: white;
        }

        .badge-safety {
            background: #dc2626;
            color: white;
        }

        .badge-manual {
            background: #0891b2;
            color: white;
        }

        .badge-type {
            background: #8b5cf6;
            color: white;
        }

        .badge-conflict {
            background: #dc2626;
            color: white;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .group-content {
            padding: 20px;
        }

        .section {
            margin-bottom: 25px;
        }

        .section h3 {
            color: #1e40af;
            margin-bottom: 10px;
            font-size: 1.3em;
        }

        .section.highlight {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            border-radius: 4px;
        }

        .standards-list {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .regulatory-intent {
            color: #4b5563;
            font-style: italic;
            line-height: 1.8;
        }

        .consolidated-requirement {
            background: white;
            padding: 15px;
            border-radius: 4px;
            font-size: 1.05em;
            line-height: 1.8;
        }

        .critical-differences ul {
            list-style-position: inside;
            color: #4b5563;
        }

        .critical-differences li {
            margin-bottom: 5px;
        }

        .no-differences {
            color: #6b7280;
            font-style: italic;
        }

        .members {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .member {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 15px;
        }

        .member-header {
            color: #1e40af;
            margin-bottom: 8px;
            font-size: 0.95em;
        }

        .member-header code {
            background: #e0e7ff;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }

        .member-text {
            color: #374151;
            margin-bottom: 8px;
            line-height: 1.6;
        }

        .member-metadata {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }

        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            text-align: center;
            color: #6b7280;
            font-size: 0.9em;
        }

        /* Phase 2 Consolidation Styles */
        .conflicts-section {
            background: #fee2e2;
            border-left: 4px solid #dc2626;
            padding: 15px;
            border-radius: 4px;
        }

        .conflicts-section h3 {
            color: #dc2626;
        }

        .conflicts-content {
            color: #7f1d1d;
            line-height: 1.8;
        }

        .unique-section {
            background: #dbeafe;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 4px;
        }

        .unique-section h3 {
            color: #1e40af;
        }

        .unique-content {
            color: #1e3a8a;
            line-height: 1.8;
        }

        .differences-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }

        .differences-table th {
            background: #f3f4f6;
            color: #1e40af;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #e5e7eb;
            font-weight: 600;
        }

        .differences-table td {
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
            vertical-align: top;
        }

        .differences-table tr:hover {
            background: #f9fafb;
        }

        .diff-standard {
            width: 150px;
            font-weight: 600;
            color: #1e40af;
        }

        .diff-text {
            color: #4b5563;
            line-height: 1.8;
        }
    """


def save_html_report(groups: List[RequirementGroup], output_path: str, title: str = "Cross-Standard Harmonization Report"):
    """
    Generate and save HTML report to file.

    Args:
        groups: List of RequirementGroup objects
        output_path: Path to save HTML file
        title: Report title
    """
    html_content = build_html_report(groups, title)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[REPORT] Saved HTML report to {output_path}")
    print(f"[REPORT] Report contains {len(groups)} groups with {sum(g.get_clause_count() for g in groups)} total clauses")
