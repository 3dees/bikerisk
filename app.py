"""
Streamlit UI for E-Bike Standards Requirement Extractor.
"""
import streamlit as st
import requests
import pandas as pd
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from consolidate_smart_ai import consolidate_with_smart_ai
from project_storage import (
    save_project,
    load_project,
    list_saved_projects,
    delete_project,
    auto_save_project,
    format_project_display
)

# MUST be the first Streamlit command - at module level, not in a function
st.set_page_config(
    page_title="E-Bike Standards Extractor",
    page_icon="🚴",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"


def render_feedback_widget():
    """Render feedback collection widget in sidebar."""
    import json  # noqa: F401 (used via alias in write)
    from datetime import datetime as _dt
    import streamlit as st

    st.divider()
    st.markdown("### 💬 Feedback")

    with st.expander("📝 Send Feedback", expanded=False):
        feedback_type = st.selectbox(
            "Type",
            ["🐛 Bug Report", "💡 Feature Request", "💬 General Feedback", "⭐ Rating"],
            key="feedback_type"
        )

        rating = None
        if feedback_type == "⭐ Rating":
            rating = st.slider(
                "How satisfied are you?",
                1, 5, 3,
                key="rating",
                help="1 = Very Unsatisfied, 5 = Very Satisfied"
            )

        feedback_text = st.text_area(
            "Your feedback:",
            placeholder="Tell us what you think...",
            height=100,
            key="feedback_text"
        )

        current_page = st.selectbox(
            "Related to:",
            ["General", "PDF Extraction", "Consolidation", "UI/Design", "Performance", "Other"],
            key="feedback_page"
        )

        contact = st.text_input(
            "Email (optional):",
            placeholder="your.email@company.com",
            key="feedback_email"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📤 Send Feedback", type="primary", use_container_width=True):
                if feedback_text.strip():
                    feedback_data = {
                        "timestamp": _dt.now().isoformat(),
                        "type": feedback_type,
                        "page": current_page,
                        "text": feedback_text,
                        "email": contact if contact else "anonymous",
                        "rating": rating,
                        "session_id": st.session_state.get('session_id', 'unknown')
                    }
                    try:
                        with open('feedback.jsonl', 'a', encoding='utf-8') as f:
                            import json as _json
                            f.write(_json.dumps(feedback_data, ensure_ascii=False) + '\n')
                        st.success("✅ Thank you! Your feedback has been sent.")
                        st.balloons()
                        st.session_state.feedback_text = ""
                        st.session_state.feedback_email = ""
                    except Exception as e:
                        st.error(f"Failed to save feedback: {e}")
                else:
                    st.warning("Please enter some feedback first!")
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.feedback_text = ""
                st.rerun()

def main():
    st.title("🚴 E-Bike Standards Requirement Extractor")

    # API Key configuration in sidebar (persistent across tabs)
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Sources: session override > st.secrets > .env
        session_override = st.session_state.get('session_api_key')
        secret_key = None
        try:
            # st.secrets is available locally (empty) and on Streamlit Cloud
            secret_key = st.secrets.get('ANTHROPIC_API_KEY', None)
        except Exception:
            secret_key = None
        env_key = os.getenv('ANTHROPIC_API_KEY')

        active_key = None
        active_source = None
        if session_override:
            active_key = session_override
            active_source = "Session override"
        elif secret_key:
            active_key = secret_key
            active_source = "st.secrets"
        elif env_key:
            active_key = env_key
            active_source = ".env"

        # Display current source and allow override
        if active_key:
            st.success(f"✅ API key detected from {active_source}")
            if st.checkbox("Override Anthropic key for this session"):
                new_key = st.text_input(
                    "Anthropic API Key",
                    value="",
                    type="password",
                    help="Use a different key without changing your .env or st.secrets"
                )
                if new_key:
                    st.session_state['session_api_key'] = new_key
                    active_key = new_key
                    active_source = "Session override"
                    st.success("✅ Using session override")
        else:
            st.warning("No Anthropic API key detected.")
            new_key = st.text_input(
                "Anthropic API Key",
                value="",
                type="password",
                help="Set a key for this session (recommended for first run)"
            )
            if new_key:
                st.session_state['session_api_key'] = new_key
                active_key = new_key
                active_source = "Session override"
                st.success("✅ API key saved for this session")

        # Persist back to legacy session key name for downstream functions
        if active_key:
            st.session_state['anthropic_api_key'] = active_key

        # .env helper note
        if os.path.exists('.env'):
            st.caption("Using .env file if present. You can also configure secrets in Streamlit Cloud.")
        else:
            with st.expander("How to set up a local .env", expanded=False):
                st.code("""# .env
ANTHROPIC_API_KEY=sk-ant-...""", language="bash")

        # Advanced network settings note
        with st.expander("Advanced network settings", expanded=False):
            st.markdown("If you're behind a corporate proxy, you may need to bypass for Anthropic:")
            st.code("os.environ['NO_PROXY'] = '*.anthropic.com'", language="python")

        # ========================================
        # PROJECT MANAGEMENT SECTION
        # ========================================
        st.divider()
        st.header("📂 Project Management")

        # Load existing project dropdown
        projects = list_saved_projects()
        if projects:
            project_options = ["-- New Project --"] + projects

            selected_idx = st.selectbox(
                "Load Project:",
                options=range(len(project_options)),
                format_func=lambda i: project_options[i]['name'] if i > 0 else "-- New Project --",
                key="project_selector"
            )

            if selected_idx > 0:
                selected_project = project_options[selected_idx]
                if st.button("📂 Load Project", use_container_width=True):
                    if load_project(selected_project['id']):
                        st.rerun()

        # Current project management
        if 'current_project' in st.session_state:
            st.divider()
            st.markdown("### 💾 Current Project")

            project_name = st.text_input(
                "Project Name:",
                value=st.session_state.current_project['name'],
                key="project_name_input"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", use_container_width=True, type="primary"):
                    save_project(project_name)
            with col2:
                if st.button("💾 Save As...", use_container_width=True):
                    new_name = st.text_input("New project name:", key="save_as_name")
                    if new_name:
                        save_project(new_name)

            # Last saved indicator
            if 'last_saved' in st.session_state:
                st.caption(f"✓ Saved at {st.session_state.last_saved}")

            # Delete project button
            if st.button("🗑️ Delete Project", use_container_width=True):
                if st.checkbox("⚠️ Confirm deletion", key="confirm_delete"):
                    if delete_project(st.session_state.current_project['id']):
                        del st.session_state.current_project
                        st.success("Project deleted")
                        st.rerun()

        # Save button even if no current project (for first-time save)
        elif 'smart_consolidation' in st.session_state:
            st.divider()
            st.markdown("### 💾 Save Results")
            new_project_name = st.text_input(
                "Project Name:",
                value=f"Project_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key="new_project_name"
            )
            if st.button("💾 Save New Project", use_container_width=True, type="primary"):
                save_project(new_project_name)

        # Feedback widget at end of sidebar
        render_feedback_widget()

    # Create tabs with dynamic selection
    if 'switch_to_consolidation' in st.session_state and st.session_state.switch_to_consolidation:
        default_tab = 1
        st.session_state.switch_to_consolidation = False
    else:
        default_tab = 0
    
    tabs = st.tabs(["📄 Extract from PDFs", "🔗 Consolidate Requirements"])
    
    # Render tabs
    with tabs[0]:
        render_extraction_tab()
    
    with tabs[1]:
        render_consolidation_tab()


def render_extraction_tab():
    """Tab 1: PDF extraction - upload in main area"""
    st.markdown("""
    Extract instruction/manual requirements from e-bike standards and regulations.
    Upload a PDF to analyze its manual/instruction requirements.
    """)

    # Check API health
    if not check_api_health():
        st.error("⚠️ Cannot connect to backend API. Please ensure the FastAPI server is running on port 8000.")
        st.code("python main.py", language="bash")
        return

    # File upload in main area (like consolidation tab)
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Upload PDF Standard",
        type=['pdf'],
        help="Upload a PDF containing e-bike standards or regulations",
        key="pdf_uploader"
    )

    if uploaded_file:
        standard_name = st.text_input(
            "Standard Name (optional)",
            placeholder="e.g., EN 15194, 16 CFR Part 1512",
            help="Name of the standard/regulation being analyzed"
        )

        # Always use AI mode
        mode_value = "ai"
        custom_section_name = None

        process_button = st.button("🔍 Process Document", type="primary")

        if process_button:
            process_document(uploaded_file, standard_name, None, "ai")

    # Show existing results if available
    if 'job_id' in st.session_state:
        display_results(st.session_state.job_id)


# ============================================================================
# CONSOLIDATION TAB HELPER FUNCTIONS
# ============================================================================

def normalize_column_names(df):
    """
    Normalize column names to match expected format.
    Handles both PDF extractor output and direct spreadsheet uploads.
    
    Expected columns after normalization:
    - 'Requirement (Clause)' or 'Description'
    - 'Standard/ Regulation' or 'Standard/Reg'
    - 'Clause ID' or 'Clause/Requirement'
    """
    # Create mapping of possible column names
    column_mapping = {
        # Description/Requirement column
        'Description': 'Requirement (Clause)',
        'Requirement': 'Requirement (Clause)',
        'Requirement Text': 'Requirement (Clause)',
        'Text': 'Requirement (Clause)',
        
        # Standard column
        'Standard/Reg': 'Standard/ Regulation',
        'Standard': 'Standard/ Regulation',
        'Regulation': 'Standard/ Regulation',
        'Standard Name': 'Standard/ Regulation',
        
        # Clause column
        'Clause/Requirement': 'Clause ID',
        'Clause': 'Clause ID',
        'Clause Number': 'Clause ID',
        'Section': 'Clause ID',
    }
    
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


def merge_dataframes(df1, df2):
    """
    Merge two requirement dataframes, handling column mismatches.
    
    Args:
        df1: Original dataframe
        df2: New dataframe to append
    
    Returns:
        Merged dataframe with normalized columns
    """
    # Normalize both dataframes
    df1_norm = normalize_column_names(df1)
    df2_norm = normalize_column_names(df2)
    
    # Get all columns from both
    all_columns = set(df1_norm.columns) | set(df2_norm.columns)
    
    # Add missing columns with NaN
    for col in all_columns:
        if col not in df1_norm.columns:
            df1_norm[col] = pd.NA
        if col not in df2_norm.columns:
            df2_norm[col] = pd.NA
    
    # Concatenate
    merged = pd.concat([df1_norm, df2_norm], ignore_index=True)
    
    return merged


def save_data_state():
    """Save current data and consolidation state to history for undo."""
    if 'consolidation_df' not in st.session_state:
        return
    
    # Create state snapshot
    state = {
        'df': st.session_state.consolidation_df.copy(),
        'consolidation': st.session_state.get('smart_consolidation'),
        'accepted': st.session_state.accepted_groups.copy(),
        'rejected': st.session_state.rejected_groups.copy(),
        'edited': st.session_state.edited_groups.copy(),
        'removed': {k: v.copy() for k, v in st.session_state.removed_requirements.items()},
        'modified': st.session_state.modified_groups.copy(),
    }
    
    # Add to history (limit to last 5 states)
    if 'data_history' not in st.session_state:
        st.session_state.data_history = []
    
    st.session_state.data_history.append(state)
    if len(st.session_state.data_history) > 5:
        st.session_state.data_history.pop(0)
    
    st.session_state.show_undo = True


def restore_previous_state():
    """Restore previous data state from history."""
    if not st.session_state.get('data_history'):
        return False
    
    # Pop last state
    state = st.session_state.data_history.pop()
    
    # Restore all components
    st.session_state.consolidation_df = state['df']
    st.session_state.smart_consolidation = state['consolidation']
    st.session_state.accepted_groups = state['accepted']
    st.session_state.rejected_groups = state['rejected']
    st.session_state.edited_groups = state['edited']
    st.session_state.removed_requirements = state['removed']
    st.session_state.modified_groups = state['modified']
    
    # Hide undo if no more history
    if not st.session_state.data_history:
        st.session_state.show_undo = False
    
    return True


def format_requirement_display(req_row, max_length=300):
    """
    Helper to format a requirement for display.

    Args:
        req_row: DataFrame row containing requirement data
        max_length: Maximum length for text display (default 300)

    Returns:
        dict with keys: text, standard, clause
    """
    req_text = req_row.get('Requirement (Clause)', req_row.get('Description', ''))
    standard = req_row.get('Standard/ Regulation', req_row.get('Standard/Reg', ''))
    clause = req_row.get('Clause ID', req_row.get('Clause/Requirement', ''))

    # Convert to string and handle NaN
    text_str = str(req_text) if pd.notna(req_text) else '[Empty]'
    standard_str = str(standard) if pd.notna(standard) else '[Unknown]'
    clause_str = str(clause) if pd.notna(clause) else '[Unknown]'

    # Truncate text if needed
    if len(text_str) > max_length:
        text_str = text_str[:max_length] + '...'

    return {
        'text': text_str,
        'standard': standard_str,
        'clause': clause_str
    }


def generate_export_data(result, session_state):
    """
    Generate export data from consolidation results.

    Args:
        result: Consolidation result dict with 'groups'
        session_state: Streamlit session state

    Returns:
        pandas DataFrame ready for export
    """
    export_data = []
    for group in result['groups']:
        # Determine status
        if group.group_id in session_state.accepted_groups:
            status = "ACCEPTED"
        elif group.group_id in session_state.rejected_groups:
            status = "REJECTED"
        else:
            status = "PENDING"

        # Get edited text if exists
        core_req = session_state.edited_groups.get(group.group_id, group.core_requirement)

        # Get active requirements (excluding removed ones)
        removed = session_state.removed_requirements.get(group.group_id, set())
        active_reqs = [idx for idx in group.requirement_indices if idx not in removed]

        export_data.append({
            'Status': status,
            'Group ID': group.group_id + 1,
            'Topic': group.topic,
            'Regulatory Intent': group.regulatory_intent,
            'Core Requirement': core_req,
            'Applies To Standards': ', '.join(group.applies_to_standards),
            'Critical Differences': '; '.join(group.critical_differences),
            'Consolidation Potential': f"{group.consolidation_potential:.0%}",
            'Requirement Count': len(active_reqs),
            'Original Indices': ', '.join(map(str, active_reqs)),
            'Removed Indices': ', '.join(map(str, removed)) if removed else 'None',
            'Modified': 'Yes' if group.group_id in session_state.modified_groups else 'No'
        })

    export_df = pd.DataFrame(export_data)

    # Convert all columns to string to prevent Arrow warnings
    for col in export_df.columns:
        export_df[col] = export_df[col].astype(str)

    return export_df


def render_consolidation_tab():
    """Tab 2: Smart AI Consolidation"""
    
    # Initialize session state for tracking actions
    if 'accepted_groups' not in st.session_state:
        st.session_state.accepted_groups = set()
    if 'rejected_groups' not in st.session_state:
        st.session_state.rejected_groups = set()
    if 'edited_groups' not in st.session_state:
        st.session_state.edited_groups = {}  # group_id -> edited text
    if 'removed_requirements' not in st.session_state:
        st.session_state.removed_requirements = {}  # group_id -> set of removed indices
    if 'show_all_groups' not in st.session_state:
        st.session_state.show_all_groups = False
    if 'modified_groups' not in st.session_state:
        st.session_state.modified_groups = set()  # Groups with removed/restored requirements
    
    # Data history for undo functionality
    if 'data_history' not in st.session_state:
        st.session_state.data_history = []  # Stack: [(df, consolidation_result), ...]
    if 'show_undo' not in st.session_state:
        st.session_state.show_undo = False
    
    st.markdown("""
    Upload a spreadsheet with requirements from multiple standards.
    
    **Claude will analyze and group requirements by regulatory intent** - focusing on what they're 
    trying to achieve, not just text similarity.
    """)

    # Check if API key is configured
    if not st.session_state.get('anthropic_api_key'):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to use AI consolidation.")
        return

    st.divider()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Requirements Spreadsheet",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing requirements",
        key="consolidation_uploader"
    )
    
    # Check if data was passed from extraction tab (only if no file uploaded)
    if 'consolidation_df' in st.session_state and uploaded_file is None:
        st.info("📊 Using data from extraction. You can also upload a different file above.")
        df = st.session_state.consolidation_df
        
        st.success(f"✅ Loaded {len(df)} requirements from extraction")
        
        # Show preview
        with st.expander(f"📋 Data Preview (first 10 rows)"):
            st.dataframe(df.head(10))
        
        # Add clear button to allow fresh upload
        if st.button("🗑️ Clear Data (Upload Different File)", use_container_width=True):
            if 'consolidation_df' in st.session_state:
                del st.session_state.consolidation_df
            st.rerun()

    if uploaded_file:
        # Read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # For Excel, let user select sheet if multiple exist
                excel_file = pd.ExcelFile(uploaded_file)
                if len(excel_file.sheet_names) > 1:
                    sheet_name = st.selectbox("Select Sheet", excel_file.sheet_names)
                else:
                    sheet_name = excel_file.sheet_names[0]

                # Read with header detection
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)

                # Find the header row
                header_row = 0
