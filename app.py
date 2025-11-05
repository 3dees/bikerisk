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


def main():
    st.title("🚴 E-Bike Standards Requirement Extractor")

    # API Key configuration in sidebar (persistent across tabs)
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Initialize API key in session state if not exists
        # Try to load from environment variable first
        if 'anthropic_api_key' not in st.session_state:
            env_key = os.getenv('ANTHROPIC_API_KEY', '')
            st.session_state.anthropic_api_key = env_key

        # Only show input if no API key is loaded from env
        env_key = os.getenv('ANTHROPIC_API_KEY')
        if env_key:
            st.success("✅ API key loaded from .env file")
            st.session_state.anthropic_api_key = env_key
            # Show option to override
            if st.checkbox("Use different API key"):
                api_key = st.text_input(
                    "Anthropic API Key",
                    value="",
                    type="password",
                    help="Enter a different Anthropic API key"
                )
                if api_key:
                    st.session_state.anthropic_api_key = api_key
                    st.success("✅ Using custom API key for this session")
        else:
            api_key = st.text_input(
                "Anthropic API Key",
                value=st.session_state.anthropic_api_key,
                type="password",
                help="Enter your Anthropic API key for AI-powered features"
            )

            if api_key:
                st.session_state.anthropic_api_key = api_key
                st.success("✅ API key saved for session")

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
                for i in range(min(5, len(df))):
                    row_str = ' '.join([str(x).lower() for x in df.iloc[i] if pd.notna(x)])
                    if 'requirement' in row_str or 'description' in row_str:
                        header_row = i
                        break

                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)

            # Normalize columns
            df = normalize_column_names(df)

            # FIX: Convert clause columns to string to avoid Arrow serialization warnings
            if 'Clause/Requirement' in df.columns:
                df['Clause/Requirement'] = df['Clause/Requirement'].astype(str)
            if 'Clause ID' in df.columns:
                df['Clause ID'] = df['Clause ID'].astype(str)
            if 'Clause' in df.columns:
                df['Clause'] = df['Clause'].astype(str)

            st.success(f"✅ Loaded {len(df)} rows")

            # ADD SEARCH FOR CONSOLIDATION DATA
            search_term = st.text_input(
                "🔍 Search Requirements",
                placeholder="Search before consolidating...",
                help="Filter requirements by keyword",
                key="consolidation_search"
            )
            
            if search_term:
                # Determine which columns to search
                search_cols = ['Requirement (Clause)', 'Description', 'Standard/ Regulation', 'Standard/Reg']
                available_cols = [col for col in search_cols if col in df.columns]
                
                # Build search mask
                mask = pd.Series([False] * len(df))
                for col in available_cols:
                    mask = mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
                
                df_filtered = df[mask]
                
                if len(df_filtered) == 0:
                    st.warning(f"No results found for '{search_term}'")
                    df_filtered = df
                else:
                    st.info(f"Found {len(df_filtered)} of {len(df)} requirements")
            else:
                df_filtered = df

            # Show preview with filtered data
            with st.expander(f"📋 Data Preview (first 10 rows)"):
                st.dataframe(df_filtered.head(10))

            # Store FILTERED data in session state
            st.session_state.consolidation_df = df_filtered

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

    # ========================================
    # DATA MANAGEMENT: Add More Data & Delete Standards
    # ========================================
    if 'consolidation_df' in st.session_state:
        df = st.session_state.consolidation_df
        
        st.divider()
        
        # Show undo button if changes were made
        if st.session_state.get('show_undo'):
            if st.button("↩️ Undo Last Change", type="secondary", use_container_width=True):
                if restore_previous_state():
                    auto_save_project()
                    st.success("✅ Previous state restored!")
                    st.rerun()
                else:
                    st.error("No previous state to restore")
        
        with st.expander("🔧 Manage Dataset", expanded=False):
            tab_add, tab_delete = st.tabs(["➕ Add More Data", "🗑️ Delete Standards"])
            
            # ===== ADD MORE DATA TAB =====
            with tab_add:
                st.markdown("**Upload additional requirements to merge with current dataset**")
                st.caption("Column names will be automatically matched")
                
                add_file = st.file_uploader(
                    "Additional Requirements File",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload spreadsheet with additional requirements",
                    key="add_data_uploader"
                )
                
                if add_file:
                    try:
                        # Read new file
                        if add_file.name.endswith('.csv'):
                            new_df = pd.read_csv(add_file)
                        else:
                            excel_file = pd.ExcelFile(add_file)
                            if len(excel_file.sheet_names) > 1:
                                sheet_name = st.selectbox("Select Sheet", excel_file.sheet_names, key="add_sheet")
                            else:
                                sheet_name = excel_file.sheet_names[0]
                            
                            new_df = pd.read_excel(add_file, sheet_name=sheet_name, header=None)
                            
                            # Find header
                            header_row = 0
                            for i in range(min(5, len(new_df))):
                                row_str = ' '.join([str(x).lower() for x in new_df.iloc[i] if pd.notna(x)])
                                if 'requirement' in row_str or 'description' in row_str:
                                    header_row = i
                                    break
                            
                            new_df = pd.read_excel(add_file, sheet_name=sheet_name, header=header_row)
                        
                        # Normalize columns
                        new_df = normalize_column_names(new_df)
                        
                        st.success(f"✅ Loaded {len(new_df)} new requirements")
                        
                        # Preview
                        with st.expander("Preview New Data"):
                            st.dataframe(new_df.head(10))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Merge into Dataset", type="primary", use_container_width=True):
                                # Save current state for undo
                                save_data_state()
                                
                                # Merge dataframes
                                merged_df = merge_dataframes(df, new_df)
                                st.session_state.consolidation_df = merged_df
                                
                                # Clear consolidation results
                                if 'smart_consolidation' in st.session_state:
                                    del st.session_state.smart_consolidation
                                
                                # Reset tracking
                                st.session_state.accepted_groups = set()
                                st.session_state.rejected_groups = set()
                                st.session_state.edited_groups = {}
                                st.session_state.removed_requirements = {}
                                st.session_state.modified_groups = set()
                                
                                # Auto-save
                                auto_save_project()
                                
                                st.success(f"✅ Merged! Total requirements: {len(merged_df)}")
                                st.info("🔄 Please re-run the analysis to include new data")
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ Cancel", use_container_width=True):
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error loading file: {e}")
            
            # ===== DELETE STANDARDS TAB =====
            with tab_delete:
                st.markdown("**Remove specific standards from current dataset**")
                
                # Get unique standards
                standard_col = 'Standard/ Regulation' if 'Standard/ Regulation' in df.columns else 'Standard/Reg'
                if standard_col in df.columns:
                    standards = sorted(df[standard_col].dropna().unique().tolist())
                    
                    if standards:
                        # Count requirements per standard
                        standard_counts = df[standard_col].value_counts().to_dict()
                        
                        # Format options with counts
                        standard_options = [f"{std} ({standard_counts.get(std, 0)} requirements)" for std in standards]
                        
                        to_remove = st.multiselect(
                            "Select Standards to Remove:",
                            options=standards,
                            format_func=lambda x: f"{x} ({standard_counts.get(x, 0)} requirements)"
                        )
                        
                        if to_remove:
                            # Calculate impact
                            rows_to_remove = df[df[standard_col].isin(to_remove)]
                            st.warning(f"⚠️ This will remove {len(rows_to_remove)} requirements from {len(to_remove)} standard(s)")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🗑️ Remove Selected", type="primary", use_container_width=True):
                                    # Save current state for undo
                                    save_data_state()
                                    
                                    # Filter dataframe
                                    filtered_df = df[~df[standard_col].isin(to_remove)].reset_index(drop=True)
                                    st.session_state.consolidation_df = filtered_df
                                    
                                    # Clear consolidation results
                                    if 'smart_consolidation' in st.session_state:
                                        del st.session_state.smart_consolidation
                                    
                                    # Reset tracking
                                    st.session_state.accepted_groups = set()
                                    st.session_state.rejected_groups = set()
                                    st.session_state.edited_groups = {}
                                    st.session_state.removed_requirements = {}
                                    st.session_state.modified_groups = set()
                                    
                                    # Auto-save
                                    auto_save_project()
                                    
                                    st.success(f"✅ Removed {len(to_remove)} standard(s). Remaining: {len(filtered_df)} requirements")
                                    st.info("🔄 Please re-run the analysis with updated data")
                                    st.rerun()
                            
                            with col2:
                                if st.button("❌ Cancel", use_container_width=True):
                                    st.rerun()
                    else:
                        st.info("No standards found in dataset")
                else:
                    st.error("❌ Standard column not found in dataset")

        st.divider()
        
        # Settings
        col1, col2 = st.columns(2)
        
        with col1:
            min_group_size = st.slider(
                "Minimum Group Size",
                min_value=2,
                max_value=8,
                value=3,
                help="Minimum number of requirements per group"
            )
        
        with col2:
            max_group_size = st.slider(
                "Maximum Group Size",
                min_value=8,
                max_value=20,
                value=12,
                help="Maximum number of requirements per group"
            )
        
        # Show estimated processing time
        total_reqs = len(df)
        if total_reqs <= 150:
            estimate = "3-8 minutes"
            batches_info = ""
        else:
            batches = (total_reqs + 149) // 150
            estimate = f"{batches * 3}-{batches * 8} minutes"
            batches_info = f" ({batches} batches)"
        
        st.info(f"⏱️ Estimated processing time: {estimate}{batches_info}")
        
        # Run consolidation button
        if st.button("🧠 Analyze with Smart AI", type="primary"):
            with st.spinner("🤖 Claude is analyzing your requirements by regulatory intent..."):
                try:
                    from consolidate_smart_ai import consolidate_with_smart_ai
                    
                    result = consolidate_with_smart_ai(
                        df,
                        st.session_state.anthropic_api_key,
                        min_group_size=min_group_size,
                        max_group_size=max_group_size
                    )
                    
                    # Store results
                    st.session_state.smart_consolidation = result
                    
                    # Reset tracking when new analysis is run
                    st.session_state.accepted_groups = set()
                    st.session_state.rejected_groups = set()
                    st.session_state.edited_groups = {}
                    st.session_state.removed_requirements = {}
                    st.session_state.modified_groups = set()
                    st.session_state.show_all_groups = False
                    
                    st.success(f"✅ Analysis complete!")
                    
                    # Show stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Requirements", result['total_requirements'])
                    with col2:
                        st.metric("Groups Created", len(result['groups']))
                    with col3:
                        st.metric("Ungrouped", result['ungrouped_count'])
                    
                except Exception as e:
                    st.error(f"❌ Error during consolidation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display results
        if 'smart_consolidation' in st.session_state:
            result = st.session_state.smart_consolidation
            
            st.divider()
            st.subheader("📊 Consolidation Results")
            
            # Analysis notes
            if result.get('analysis_notes'):
                st.info(f"**Analysis:** {result['analysis_notes']}")
            
            # Add sorting controls
            col1, col2 = st.columns([2, 1])
            with col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=["Group Number", "Similarity Score (High to Low)", "Similarity Score (Low to High)", "Topic"],
                    key="sort_consolidations"
                )
            
            # Sort the groups based on selection
            if sort_by == "Similarity Score (High to Low)":
                sorted_groups = sorted(result['groups'], key=lambda g: g.consolidation_potential, reverse=True)
            elif sort_by == "Similarity Score (Low to High)":
                sorted_groups = sorted(result['groups'], key=lambda g: g.consolidation_potential)
            elif sort_by == "Topic":
                sorted_groups = sorted(result['groups'], key=lambda g: g.topic.lower())
            else:  # Group Number
                sorted_groups = sorted(result['groups'], key=lambda g: g.group_id)
            
            # Performance optimization: limit visible groups initially for large datasets
            if len(sorted_groups) > 20 and not st.session_state.show_all_groups:
                st.warning(f"⚡ Showing first 20 groups (out of {len(sorted_groups)} total) for better performance")
                if st.button("📜 Show All Groups"):
                    st.session_state.show_all_groups = True
                    st.rerun()
                groups_to_display = sorted_groups[:20]
            else:
                groups_to_display = sorted_groups
            
            # Display each group
            for group in groups_to_display:
                # Color code by consolidation potential
                if group.consolidation_potential >= 0.85:
                    status_emoji = "🟢"
                    status_text = "High Confidence"
                elif group.consolidation_potential >= 0.7:
                    status_emoji = "🟡"
                    status_text = "Medium Confidence"
                else:
                    status_emoji = "🟠"
                    status_text = "Review Needed"
                
                # Check if this group has been accepted/rejected
                is_accepted = group.group_id in st.session_state.accepted_groups
                is_rejected = group.group_id in st.session_state.rejected_groups
                is_modified = group.group_id in st.session_state.modified_groups
                
                # Add status badge to expander title
                status_badge = ""
                if is_accepted:
                    status_badge = " ✅ ACCEPTED"
                elif is_rejected:
                    status_badge = " ❌ REJECTED"
                if is_modified:
                    status_badge += " ⚠️ MODIFIED"
                
                with st.expander(
                    f"{status_emoji} **Group {group.group_id + 1}: {group.topic}** "
                    f"({len(group.requirement_indices)} requirements) - "
                    f"{group.consolidation_potential:.0%} match{status_badge}"
                ):
                    # Regulatory Intent
                    st.markdown("### 🎯 Regulatory Intent")
                    st.info(group.regulatory_intent)
                    
                    # Core Requirement (enhanced display with editing capability)
                    st.markdown("### 📌 Consolidated Requirement (Ready to Use)")
                    st.caption("💡 This detailed requirement can be used directly in your product manual")
                    
                    # Show warning if requirements were removed
                    if is_modified:
                        st.warning("⚠️ Requirements were removed from this group. Consider reviewing the consolidated requirement for accuracy.")
                    
                    # Check if we're editing this group
                    if st.session_state.get(f'editing_{group.group_id}', False):
                        # Show text area for editing
                        edited_text = st.text_area(
                            "Edit Core Requirement:",
                            value=st.session_state.edited_groups.get(group.group_id, group.core_requirement),
                            height=300,
                            key=f"edit_area_{group.group_id}"
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("💾 Save Changes", key=f"save_{group.group_id}"):
                                st.session_state.edited_groups[group.group_id] = edited_text
                                st.session_state[f'editing_{group.group_id}'] = False
                                auto_save_project()
                                st.success("Changes saved!")
                                st.rerun()
                        with col2:
                            if st.button("🔄 Revert to Original", key=f"revert_{group.group_id}"):
                                if group.group_id in st.session_state.edited_groups:
                                    del st.session_state.edited_groups[group.group_id]
                                st.session_state[f'editing_{group.group_id}'] = False
                                st.success("Reverted to original!")
                                st.rerun()
                        with col3:
                            if st.button("❌ Cancel", key=f"cancel_{group.group_id}"):
                                st.session_state[f'editing_{group.group_id}'] = False
                                st.rerun()
                    else:
                        # Display the requirement (use edited version if it exists)
                        display_text = st.session_state.edited_groups.get(group.group_id, group.core_requirement)
                        st.markdown("""
                        <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; color: #155724;">
                        <strong>""" + display_text.replace('\n', '<br>') + """</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Standards Covered
                    st.markdown("### 📋 Applies To")
                    st.write(", ".join(group.applies_to_standards))
                    
                    # Critical Differences
                    if group.critical_differences:
                        st.markdown("### ⚠️ Critical Differences to Preserve")
                        for diff in group.critical_differences:
                            st.markdown(f"- {diff}")
                    
                    # Reasoning
                    st.markdown("### 💡 Why These Were Grouped")
                    st.markdown(group.reasoning)
                    
                    # Show original requirements with management controls
                    st.markdown("---")
                    st.markdown("### 📄 Original Requirements")
                    
                    # Initialize removed set for this group if not exists
                    if group.group_id not in st.session_state.removed_requirements:
                        st.session_state.removed_requirements[group.group_id] = set()
                    
                    # Toggle removal mode
                    removal_mode = st.toggle("Enable Removal Mode", key=f"removal_mode_{group.group_id}")
                    
                    if removal_mode:
                        st.caption("⚠️ Check requirements to remove them from this group")
                    
                    # Filter out removed requirements
                    active_indices = [idx for idx in group.requirement_indices 
                                      if idx not in st.session_state.removed_requirements[group.group_id]]
                    
                    for idx in active_indices:
                        if idx < len(df):
                            req_row = df.iloc[idx]
                            req_info = format_requirement_display(req_row)

                            if removal_mode:
                                # Show checkbox when in removal mode
                                col_check, col_req = st.columns([1, 20])
                                with col_check:
                                    if st.checkbox(f"Remove requirement {idx}", key=f"check_{group.group_id}_{idx}", label_visibility="collapsed"):
                                        st.session_state.removed_requirements[group.group_id].add(idx)
                                        st.session_state.modified_groups.add(group.group_id)
                                        auto_save_project()
                                        st.rerun()
                                with col_req:
                                    st.markdown(f"**{req_info['standard']}** (Clause {req_info['clause']})")
                                    st.caption(req_info['text'])
                            else:
                                # Normal display
                                st.markdown(f"**{req_info['standard']}** (Clause {req_info['clause']})")
                                st.caption(req_info['text'])

                            st.markdown("")  # spacing
                    
                    # Show removed requirements (FIXED: using checkbox instead of nested expander)
                    removed_indices = st.session_state.removed_requirements[group.group_id]
                    if removed_indices:
                        show_removed = st.checkbox(
                            f"Show Removed Requirements ({len(removed_indices)})", 
                            key=f"show_removed_{group.group_id}"
                        )
                        if show_removed:
                            st.markdown("**🗑️ Removed Requirements:**")
                            for idx in removed_indices:
                                if idx < len(df):
                                    req_row = df.iloc[idx]
                                    req_info = format_requirement_display(req_row, max_length=100)

                                    col_req, col_restore = st.columns([10, 1])
                                    with col_req:
                                        st.caption(f"~~{req_info['standard']}: {req_info['text']}~~")
                                    with col_restore:
                                        if st.button("↩️", key=f"restore_{group.group_id}_{idx}", help="Restore to group"):
                                            st.session_state.removed_requirements[group.group_id].remove(idx)
                                            # Remove modified flag if no more removed items
                                            if not st.session_state.removed_requirements[group.group_id]:
                                                st.session_state.modified_groups.discard(group.group_id)
                                            auto_save_project()
                                            st.rerun()
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if is_accepted:
                            st.success("✅ Accepted")
                            if st.button("Undo Accept", key=f"undo_accept_{group.group_id}"):
                                st.session_state.accepted_groups.remove(group.group_id)
                                st.rerun()
                        else:
                            if st.button(f"✅ Accept", key=f"accept_smart_{group.group_id}"):
                                st.session_state.accepted_groups.add(group.group_id)
                                if group.group_id in st.session_state.rejected_groups:
                                    st.session_state.rejected_groups.remove(group.group_id)
                                auto_save_project()
                                st.rerun()
                    
                    with col2:
                        if st.button(f"✏️ Edit Core Requirement", key=f"edit_smart_{group.group_id}"):
                            st.session_state[f'editing_{group.group_id}'] = True
                            st.rerun()
                    
                    with col3:
                        if is_rejected:
                            st.warning("❌ Rejected")
                            if st.button("Undo Reject", key=f"undo_reject_{group.group_id}"):
                                st.session_state.rejected_groups.remove(group.group_id)
                                st.rerun()
                        else:
                            if st.button(f"❌ Reject", key=f"reject_smart_{group.group_id}"):
                                st.session_state.rejected_groups.add(group.group_id)
                                if group.group_id in st.session_state.accepted_groups:
                                    st.session_state.accepted_groups.remove(group.group_id)
                                auto_save_project()
                                st.rerun()
            
            # Export/Print section
            st.divider()
            
            # Show summary of actions
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accepted", len(st.session_state.accepted_groups))
            with col2:
                st.metric("Rejected", len(st.session_state.rejected_groups))
            with col3:
                st.metric("Edited", len(st.session_state.edited_groups))
            with col4:
                st.metric("Modified", len(st.session_state.modified_groups))
            
            # Generate export data ONCE (outside button clicks to avoid nested buttons)
            export_df = generate_export_data(result, st.session_state)
            csv = export_df.to_csv(index=False)

            # Generate HTML report
            html_content = generate_html_report(result, st.session_state, df)

            # Combined Export/Print buttons - DIRECT download (no nested buttons!)
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 Export Full Report (CSV)",
                    data=csv,
                    file_name="smart_consolidation_report.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

            with col2:
                st.download_button(
                    label="🖨️ Print-Friendly Report (HTML)",
                    data=html_content,
                    file_name="consolidation_report.html",
                    mime="text/html",
                    type="secondary",
                    use_container_width=True
                )


def generate_html_report(result, session_state, df):
    """Generate a print-friendly HTML report."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Consolidation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #28a745; border-bottom: 3px solid #28a745; padding-bottom: 10px; }
            h2 { color: #155724; margin-top: 30px; page-break-before: always; }
            h3 { color: #495057; margin-top: 20px; }
            .group { border: 2px solid #d4edda; padding: 20px; margin: 20px 0; background-color: #f8f9fa; }
            .accepted { border-left: 5px solid #28a745; }
            .rejected { border-left: 5px solid #dc3545; }
            .pending { border-left: 5px solid #ffc107; }
            .core-req { background-color: #d4edda; padding: 15px; border-radius: 5px; color: #155724; font-weight: bold; }
            .metadata { color: #6c757d; font-size: 0.9em; }
            .requirement { margin: 10px 0; padding: 10px; background-color: white; }
            @media print {
                .group { page-break-inside: avoid; }
                h2 { page-break-before: always; }
            }
        </style>
    </head>
    <body>
        <h1>📊 E-Bike Standards Consolidation Report</h1>
        <p class="metadata">Generated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        <p class="metadata">Total Requirements: """ + str(result['total_requirements']) + """</p>
        <p class="metadata">Groups Created: """ + str(len(result['groups'])) + """</p>
    """
    
    # Add each group
    for group in result['groups']:
        # Determine status
        if group.group_id in session_state.accepted_groups:
            status = "ACCEPTED"
            status_class = "accepted"
        elif group.group_id in session_state.rejected_groups:
            status = "REJECTED"
            status_class = "rejected"
        else:
            status = "PENDING"
            status_class = "pending"
        
        # Get edited text if exists
        core_req = session_state.edited_groups.get(group.group_id, group.core_requirement)
        
        # Get active requirements
        removed = session_state.removed_requirements.get(group.group_id, set())
        active_reqs = [idx for idx in group.requirement_indices if idx not in removed]
        
        html += f"""
        <div class="group {status_class}">
            <h2>Group {group.group_id + 1}: {group.topic}</h2>
            <p class="metadata">Status: <strong>{status}</strong> | Similarity: {group.consolidation_potential:.0%} | Requirements: {len(active_reqs)}</p>
            
            <h3>🎯 Regulatory Intent</h3>
            <p>{group.regulatory_intent}</p>
            
            <h3>📌 Consolidated Requirement</h3>
            <div class="core-req">{core_req.replace(chr(10), '<br>')}</div>
            
            <h3>📋 Applies To</h3>
            <p>{', '.join(group.applies_to_standards)}</p>
        """
        
        if group.critical_differences:
            html += "<h3>⚠️ Critical Differences</h3><ul>"
            for diff in group.critical_differences:
                html += f"<li>{diff}</li>"
            html += "</ul>"
        
        html += "<h3>📄 Original Requirements</h3>"
        for idx in active_reqs:
            if idx < len(df):
                req_row = df.iloc[idx]
                req_text = req_row.get('Requirement (Clause)', req_row.get('Description', ''))
                standard = req_row.get('Standard/ Regulation', req_row.get('Standard/Reg', ''))
                clause = req_row.get('Clause ID', req_row.get('Clause/Requirement', ''))
                
                html += f"""
                <div class="requirement">
                    <strong>{standard}</strong> (Clause {clause})<br>
                    <span class="metadata">{str(req_text)[:500]}{'...' if len(str(req_text)) > 500 else ''}</span>
                </div>
                """
        
        html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    return html


def check_api_health():
    """Check if FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False


def process_document(uploaded_file, standard_name, custom_section_name, extraction_mode="ai"):
    """Process uploaded document through the API."""

    # Fun facts to show during AI processing
    fun_facts = [
        "🐙 Octopuses have three hearts and blue blood!",
        "🍯 Honey never spoils - archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible!",
        "🦒 Giraffes and humans have the same number of neck vertebrae: seven!",
        "🌊 There's more computing power in your phone than NASA used to put humans on the moon!",
        "🧠 Your brain uses 20% of your body's energy but only weighs 2% of your body mass!",
        "⚡ Bananas are slightly radioactive due to their potassium content!",
        "🌍 The Earth's core is as hot as the surface of the sun - about 10,800°F!",
        "🦈 Sharks have been around longer than trees - by about 50 million years!",
        "👁️ Mantis shrimp can see colors we can't even imagine - they have 16 color receptors vs our 3!",
        "🚀 In space, astronauts can grow up to 2 inches taller due to lack of gravity!",
        "🐌 A snail can sleep for 3 years straight!",
        "💎 It rains diamonds on Jupiter and Saturn!",
        "🦋 Butterflies can taste with their feet!",
        "🌙 The footprints on the moon will last for millions of years - there's no wind to erase them!",
        "🧬 Humans share 60% of their DNA with bananas!",
        "🐝 Bees can recognize human faces!",
        "⏰ A day on Venus is longer than a year on Venus!",
        "🌊 The Pacific Ocean is wider than the moon!",
        "🦠 There are more bacterial cells in your body than human cells!",
        "🎵 The loudest sound ever recorded was the Krakatoa volcano - heard 3,000 miles away!",
    ]

    import random
    spinner_text = random.choice(fun_facts) if extraction_mode == "ai" else "⚙️ Using rule-based extraction..."

    with st.spinner(spinner_text):
        try:
            # Prepare file and form data
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')
            }

            params = {
                'extraction_mode': extraction_mode
            }

            if standard_name:
                params['standard_name'] = standard_name
            if custom_section_name:
                params['custom_section_name'] = custom_section_name

            # Add API key for AI mode
            if extraction_mode == "ai" and st.session_state.get('anthropic_api_key'):
                params['api_key'] = st.session_state.anthropic_api_key

            # Call upload API
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                params=params,
                timeout=120  # Increased timeout for AI processing
            )

            if response.status_code == 200:
                result = response.json()
                st.session_state.job_id = result['job_id']

                # Show success message
                mode_emoji = "🤖" if result.get('extraction_mode') == 'ai' else "⚙️"
                st.success(f"✅ Document processed successfully using {mode_emoji} {result.get('extraction_mode', 'unknown').upper()} mode!")

                # Show extraction stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Extraction Method", result.get('extraction_method', 'unknown'))
                with col2:
                    st.metric("Confidence", result.get('extraction_confidence', 'unknown').upper())
                with col3:
                    st.metric("Total Detected", result['stats'].get('total_detected', 0))
                with col4:
                    st.metric("Classified Rows", result['stats'].get('classified_rows', 0))

            elif response.status_code == 422:
                error_detail = response.json()['detail']
                st.error(f"❌ Extraction failed: {error_detail['message']}")
                st.info(f"💡 Suggestion: {error_detail['suggestion']}")

            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The document may be too large or complex.")
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")


def display_results(job_id):
    """Display results for a job."""
    try:
        response = requests.get(f"{API_BASE_URL}/results/{job_id}")
        if response.status_code != 200:
            st.error(f"Failed to load results: {response.status_code}")
            return

        result = response.json()

        st.divider()

        # Header with metadata
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"📄 Results: {result['filename']}")
            st.caption(f"Standard: {result['standard_name']}")
        with col2:
            if st.button("🗑️ Clear Results"):
                del st.session_state.job_id
                st.rerun()

        # Display extraction info
        if result.get('extraction_method'):
            with st.expander("ℹ️ Extraction Information"):
                st.write(f"**Method:** {result['extraction_method']}")
                st.write(f"**Confidence:** {result.get('extraction_confidence', 'unknown').upper()}")
                st.write(f"**Stats:**")
                stats = result.get('stats', {})
                st.json(stats)

        # Main results table
        st.subheader("📋 Extracted Requirements")

        rows = result.get('rows', [])
        if not rows:
            st.info("No requirements found in this document.")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Keep only the 9 schema columns
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

        # ADD SEARCH FUNCTIONALITY HERE
        st.divider()
        
        # Search box
        search_term = st.text_input(
            "🔍 Search Requirements",
            placeholder="Search in Description, Comments, Clause, Standard...",
            help="Filter results by keyword (searches across multiple columns)",
            key="extraction_search"
        )
        
        # Filter dataframe if search term provided
        if search_term:
            # Search across multiple columns
            mask = (
                df['Description'].astype(str).str.contains(search_term, case=False, na=False) |
                df['Comments'].astype(str).str.contains(search_term, case=False, na=False) |
                df['Clause/Requirement'].astype(str).str.contains(search_term, case=False, na=False) |
                df['Standard/Reg'].astype(str).str.contains(search_term, case=False, na=False)
            )
            df_filtered = df[mask]
            
            if len(df_filtered) == 0:
                st.warning(f"No results found for '{search_term}'")
                df_filtered = df  # Show all if no matches
            else:
                st.success(f"Found {len(df_filtered)} of {len(df)} requirements")
        else:
            df_filtered = df

        # Initialize edited data in session state if not exists
        if 'edited_data' not in st.session_state or st.session_state.get('current_job_id') != st.session_state.job_id:
            st.session_state.edited_data = df_filtered[display_columns].copy()
            st.session_state.current_job_id = st.session_state.job_id

        # Instructions
        st.info("💡 **Tips**: \n- Click any cell to edit\n- Add new rows using the **+** button at the bottom\n- Delete rows by clearing all cells in a row")

        # Merge functionality
        with st.expander("🔗 Merge Rows"):
            st.markdown("**Select rows to merge** (by row number)")

            # Let user select which rows to merge
            available_rows = list(range(len(st.session_state.edited_data)))
            selected_rows = st.multiselect(
                "Choose 2 or more rows to merge:",
                options=available_rows,
                format_func=lambda x: f"Row {x}: {st.session_state.edited_data.iloc[x]['Description'][:50]}..."
            )

            if len(selected_rows) >= 2:
                st.write(f"**Selected {len(selected_rows)} rows for merging**")

                # Show preview of merge
                merge_descriptions = []
                merge_clauses = []
                merge_scopes = set()
                merge_comments = []

                for idx in selected_rows:
                    row = st.session_state.edited_data.iloc[idx]
                    merge_descriptions.append(row['Description'])
                    if pd.notna(row['Clause/Requirement']):
                        merge_clauses.append(str(row['Clause/Requirement']))
                    if pd.notna(row['Requirement scope']):
                        merge_scopes.add(str(row['Requirement scope']))
                    if pd.notna(row['Comments']):
                        merge_comments.append(str(row['Comments']))

                merged_desc = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(merge_descriptions)])
                merged_clause = ", ".join(merge_clauses)
                merged_scope = ", ".join(merge_scopes)
                merged_comments = "; ".join(merge_comments)

                st.text_area("Merged Description Preview:", merged_desc, height=150)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Merge These Rows", type="primary"):
                        # Create merged row
                        first_row = st.session_state.edited_data.iloc[selected_rows[0]].copy()
                        first_row['Description'] = merged_desc
                        first_row['Clause/Requirement'] = merged_clause
                        first_row['Requirement scope'] = merged_scope
                        first_row['Comments'] = merged_comments + " (merged)"

                        # Remove selected rows and add merged row
                        df_temp = st.session_state.edited_data.copy()
                        df_temp = df_temp.drop(selected_rows)
                        df_temp = pd.concat([df_temp, pd.DataFrame([first_row])], ignore_index=True)

                        st.session_state.edited_data = df_temp
                        st.success(f"✅ Merged {len(selected_rows)} rows!")
                        st.rerun()

                with col2:
                    if st.button("❌ Cancel"):
                        st.rerun()

        # Editable DataFrame with text wrapping
        edited_df = st.data_editor(
            st.session_state.edited_data,
            use_container_width=True,
            hide_index=False,
            height=500,
            num_rows="dynamic",
            column_config={
                # Main content columns with wrapping
                "Description": st.column_config.TextColumn(
                    "Description",
                    width="large",
                    help="Full requirement text - click to edit"
                ),
                "Comments": st.column_config.TextColumn(
                    "Comments",
                    width="medium",
                    help="AI notes and context"
                ),
                # Reference columns - compact
                "Standard/Reg": st.column_config.TextColumn(
                    "Standard/Reg",
                    width="small"
                ),
                "Clause/Requirement": st.column_config.TextColumn(
                    "Clause/Requirement",
                    width="small"
                ),
            },
            key="data_editor"
        )

        # Update session state with edited data
        st.session_state.edited_data = edited_df

        st.caption(f"Total requirements: {len(edited_df)}")

        # Export button - one-click download using edited data
        st.divider()

        # Convert edited DataFrame to CSV
        csv_data = edited_df.to_csv(index=False)

        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{result['filename']}_requirements.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True
            )
        
        with col2:
            if st.button("➡️ Continue to Consolidation", type="primary", use_container_width=True):
                # Store edited data for consolidation
                st.session_state.consolidation_df = edited_df.copy()
                
                # Clear any previous consolidation results
                if 'smart_consolidation' in st.session_state:
                    del st.session_state.smart_consolidation
                
                # Reset consolidation tracking
                st.session_state.accepted_groups = set()
                st.session_state.rejected_groups = set()
                st.session_state.edited_groups = {}
                st.session_state.removed_requirements = {}
                st.session_state.modified_groups = set()
                st.session_state.show_all_groups = False
                
                # Set flag to switch tabs
                st.session_state.switch_to_consolidation = True
                
                st.success("✅ Data loaded into consolidation tab!")
                st.rerun()

    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")


if __name__ == "__main__":
    main()