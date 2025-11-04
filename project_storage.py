"""
Project storage and management for E-Bike Standards consolidation.
Handles saving/loading consolidation projects with all user actions.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional
import hashlib


# Storage directory
PROJECTS_DIR = Path("saved_projects")
PROJECTS_DIR.mkdir(exist_ok=True)

PROJECT_INDEX_FILE = PROJECTS_DIR / "project_index.json"


def generate_project_id(project_name: str) -> str:
    """Generate unique project ID from name and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_').lower()
    return f"{safe_name}_{timestamp}"


def calculate_dataframe_hash(df: pd.DataFrame) -> str:
    """Calculate hash of dataframe to detect changes."""
    return hashlib.md5(df.to_json().encode()).hexdigest()


def save_project(project_name: Optional[str] = None, auto_save: bool = False) -> bool:
    """
    Save current consolidation project to JSON file.

    Args:
        project_name: Name for the project. If None, uses existing name.
        auto_save: If True, saves silently without user feedback.

    Returns:
        bool: True if save successful, False otherwise
    """

    # Check if we have consolidation results to save
    if 'smart_consolidation' not in st.session_state:
        if not auto_save:
            st.warning("No consolidation results to save. Please run analysis first.")
        return False

    if 'consolidation_df' not in st.session_state:
        if not auto_save:
            st.warning("No source data found. Cannot save project.")
        return False

    try:
        # Determine project name
        if project_name is None:
            if 'current_project' in st.session_state:
                project_name = st.session_state.current_project['name']
            else:
                project_name = f"Project_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # Generate or reuse project ID
        if 'current_project' in st.session_state and not project_name != st.session_state.current_project['name']:
            project_id = st.session_state.current_project['id']
        else:
            project_id = generate_project_id(project_name)

        df = st.session_state.consolidation_df
        result = st.session_state.smart_consolidation

        # Extract standards from dataframe
        standards_col = 'Standard/ Regulation' if 'Standard/ Regulation' in df.columns else 'Standard/Reg'
        standards_included = df[standards_col].dropna().unique().tolist() if standards_col in df.columns else []

        # Build project data structure
        project_data = {
            "metadata": {
                "project_name": project_name,
                "project_id": project_id,
                "created_at": st.session_state.current_project['created_at'] if 'current_project' in st.session_state else datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "standards_included": [str(s) for s in standards_included],
                "total_requirements": len(df),
                "file_hash": calculate_dataframe_hash(df)
            },
            "source_data": {
                "dataframe_json": df.to_json(orient='records'),
                "filename": st.session_state.get('uploaded_filename', 'unknown.xlsx')
            },
            "consolidation_results": {
                "groups": [
                    {
                        "group_id": g.group_id,
                        "topic": g.topic,
                        "regulatory_intent": g.regulatory_intent,
                        "core_requirement": g.core_requirement,
                        "applies_to_standards": g.applies_to_standards,
                        "critical_differences": g.critical_differences,
                        "consolidation_potential": g.consolidation_potential,
                        "requirement_indices": g.requirement_indices,
                        "reasoning": g.reasoning
                    }
                    for g in result['groups']
                ],
                "ungrouped_indices": result.get('ungrouped_indices', []),
                "analysis_notes": result.get('analysis_notes', ''),
                "total_requirements": result.get('total_requirements', 0),
                "grouped_count": result.get('grouped_count', 0),
                "ungrouped_count": result.get('ungrouped_count', 0)
            },
            "user_actions": {
                "accepted_groups": list(st.session_state.get('accepted_groups', set())),
                "rejected_groups": list(st.session_state.get('rejected_groups', set())),
                "edited_groups": st.session_state.get('edited_groups', {}),
                "removed_requirements": {
                    str(k): list(v) for k, v in st.session_state.get('removed_requirements', {}).items()
                },
                "modified_groups": list(st.session_state.get('modified_groups', set()))
            }
        }

        # Save project file
        project_file = PROJECTS_DIR / f"{project_id}.json"
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        # Update project index
        update_project_index(project_id, project_name, project_data['metadata'])

        # Update session state
        st.session_state.current_project = {
            'id': project_id,
            'name': project_name,
            'created_at': project_data['metadata']['created_at'],
            'file_path': str(project_file)
        }
        st.session_state.last_saved = datetime.now().strftime("%I:%M %p")

        if not auto_save:
            st.success(f"✅ Project '{project_name}' saved successfully!")

        return True

    except Exception as e:
        if not auto_save:
            st.error(f"❌ Error saving project: {str(e)}")
        print(f"[STORAGE ERROR] Failed to save project: {e}")
        return False


def load_project(project_id: str) -> bool:
    """
    Load a saved project and restore all state.

    Args:
        project_id: ID of the project to load

    Returns:
        bool: True if load successful, False otherwise
    """

    try:
        project_file = PROJECTS_DIR / f"{project_id}.json"

        if not project_file.exists():
            st.error(f"❌ Project file not found: {project_id}")
            return False

        # Load project data
        with open(project_file, 'r') as f:
            project_data = json.load(f)

        # Restore dataframe
        df = pd.read_json(project_data['source_data']['dataframe_json'], orient='records')
        st.session_state.consolidation_df = df
        st.session_state.uploaded_filename = project_data['source_data']['filename']

        # Restore consolidation results
        from consolidate_smart_ai import ConsolidationGroup

        groups = []
        for g_data in project_data['consolidation_results']['groups']:
            group = ConsolidationGroup(
                group_id=g_data['group_id'],
                topic=g_data['topic'],
                regulatory_intent=g_data['regulatory_intent'],
                core_requirement=g_data['core_requirement'],
                applies_to_standards=g_data['applies_to_standards'],
                critical_differences=g_data['critical_differences'],
                consolidation_potential=g_data['consolidation_potential'],
                requirement_indices=g_data['requirement_indices'],
                reasoning=g_data['reasoning']
            )
            groups.append(group)

        st.session_state.smart_consolidation = {
            'groups': groups,
            'ungrouped_indices': project_data['consolidation_results']['ungrouped_indices'],
            'analysis_notes': project_data['consolidation_results']['analysis_notes'],
            'total_requirements': project_data['consolidation_results']['total_requirements'],
            'grouped_count': project_data['consolidation_results']['grouped_count'],
            'ungrouped_count': project_data['consolidation_results']['ungrouped_count']
        }

        # Restore user actions
        actions = project_data['user_actions']
        st.session_state.accepted_groups = set(actions['accepted_groups'])
        st.session_state.rejected_groups = set(actions['rejected_groups'])
        st.session_state.edited_groups = actions['edited_groups']
        st.session_state.removed_requirements = {
            int(k): set(v) for k, v in actions['removed_requirements'].items()
        }
        st.session_state.modified_groups = set(actions['modified_groups'])

        # Set current project
        st.session_state.current_project = {
            'id': project_id,
            'name': project_data['metadata']['project_name'],
            'created_at': project_data['metadata']['created_at'],
            'file_path': str(project_file)
        }

        st.success(f"✅ Loaded project: {project_data['metadata']['project_name']}")
        return True

    except Exception as e:
        st.error(f"❌ Error loading project: {str(e)}")
        print(f"[STORAGE ERROR] Failed to load project: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_saved_projects() -> List[Dict]:
    """
    Get list of all saved projects with metadata.

    Returns:
        List of dicts with project info
    """

    try:
        if not PROJECT_INDEX_FILE.exists():
            return []

        with open(PROJECT_INDEX_FILE, 'r') as f:
            index = json.load(f)

        # Sort by last modified (newest first)
        projects = sorted(
            index.get('projects', []),
            key=lambda x: x.get('last_modified', ''),
            reverse=True
        )

        return projects

    except Exception as e:
        print(f"[STORAGE ERROR] Failed to list projects: {e}")
        return []


def update_project_index(project_id: str, project_name: str, metadata: Dict):
    """Update the project index file with new/updated project."""

    try:
        # Load existing index
        if PROJECT_INDEX_FILE.exists():
            with open(PROJECT_INDEX_FILE, 'r') as f:
                index = json.load(f)
        else:
            index = {'projects': []}

        # Remove existing entry if updating
        index['projects'] = [p for p in index['projects'] if p['id'] != project_id]

        # Add new entry
        index['projects'].append({
            'id': project_id,
            'name': project_name,
            'created_at': metadata['created_at'],
            'last_modified': metadata['last_modified'],
            'standards_count': len(metadata['standards_included']),
            'requirements_count': metadata['total_requirements']
        })

        # Save index
        with open(PROJECT_INDEX_FILE, 'w') as f:
            json.dump(index, f, indent=2)

    except Exception as e:
        print(f"[STORAGE ERROR] Failed to update index: {e}")


def delete_project(project_id: str) -> bool:
    """Delete a saved project."""

    try:
        project_file = PROJECTS_DIR / f"{project_id}.json"

        if project_file.exists():
            project_file.unlink()

        # Update index
        if PROJECT_INDEX_FILE.exists():
            with open(PROJECT_INDEX_FILE, 'r') as f:
                index = json.load(f)

            index['projects'] = [p for p in index['projects'] if p['id'] != project_id]

            with open(PROJECT_INDEX_FILE, 'w') as f:
                json.dump(index, f, indent=2)

        return True

    except Exception as e:
        print(f"[STORAGE ERROR] Failed to delete project: {e}")
        return False


def auto_save_project():
    """Auto-save current project if one is loaded."""

    if 'current_project' in st.session_state:
        save_project(auto_save=True)


def format_project_display(project: Dict) -> str:
    """Format project info for display in dropdown."""

    from datetime import datetime

    try:
        modified = datetime.fromisoformat(project['last_modified'])
        date_str = modified.strftime("%b %d, %I:%M %p")
    except:
        date_str = "Unknown date"

    return f"{project['name']} ({date_str}) - {project['requirements_count']} reqs"
