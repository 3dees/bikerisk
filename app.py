"""
Streamlit UI for E-Bike Standards Requirement Extractor.
"""
import streamlit as st
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv
from consolidate_smart_ai import consolidate_with_smart_ai

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

    # Create tabs
    tab1, tab2 = st.tabs(["📄 Extract from PDFs", "🔗 Consolidate Requirements"])

    with tab1:
        render_extraction_tab()

    with tab2:
        render_consolidation_tab()


def render_extraction_tab():
    """Tab 1: PDF extraction (original functionality)"""
    st.markdown("""
    Extract instruction/manual requirements from e-bike standards and regulations.
    Upload a PDF to analyze its manual/instruction requirements.
    """)

    # Check API health
    if not check_api_health():
        st.error("⚠️ Cannot connect to backend API. Please ensure the FastAPI server is running on port 8000.")
        st.code("python main.py", language="bash")
        return

    # Sidebar for file upload
    with st.sidebar:
        st.divider()
        st.header("📁 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF containing e-bike standards or regulations",
            key="pdf_uploader"
        )

        standard_name = st.text_input(
            "Standard Name (optional)",
            placeholder="e.g., EN 15194, 16 CFR Part 1512",
            help="Name of the standard/regulation being analyzed"
        )

        # Extraction mode toggle
        extraction_mode = st.radio(
            "Extraction Mode",
            options=["🤖 AI (Recommended)", "⚙️ Rule-Based (Legacy)"],
            index=0,
            help="AI mode uses Claude for intelligent extraction. Rule-based uses pattern matching."
        )

        # Convert UI label to API value
        mode_value = "ai" if "AI" in extraction_mode else "rules"

        # Only show custom section name for rule-based mode
        custom_section_name = None
        if mode_value == "rules":
            custom_section_name = st.text_input(
                "Custom Section Name (optional)",
                placeholder="e.g., Instruction for use",
                help="Specific section name to look for in the document. Leave blank to use default patterns."
            )

        process_button = st.button("🔍 Process Document", type="primary", use_container_width=True)

    # Main content area
    if process_button and uploaded_file:
        process_document(uploaded_file, standard_name, custom_section_name, mode_value)

    # Show existing results if available
    if 'job_id' in st.session_state:
        display_results(st.session_state.job_id)


def render_consolidation_tab():
    """Tab 2: Requirements consolidation (Track 1)"""
    st.markdown("""
    Upload a spreadsheet with requirements from multiple standards.
    AI will analyze and suggest consolidations based on topic similarity and regulatory intent.
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

                # Find the header row (look for "Requirement" or "Description" in first 5 rows)
                header_row = 0
                for i in range(min(5, len(df))):
                    row_str = ' '.join([str(x).lower() for x in df.iloc[i] if pd.notna(x)])
                    if 'requirement' in row_str or 'description' in row_str:
                        header_row = i
                        break

                # Re-read with correct header
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)

            st.success(f"✅ Loaded {len(df)} rows")

            # Show preview with more details
            with st.expander(f"📋 Data Preview (showing first 10 of {len(df)} rows)"):
                st.dataframe(df.head(10))
                st.caption(f"**Total rows loaded:** {len(df)}")

            # Store in session state
            st.session_state.consolidation_df = df

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

    # If data is loaded, show consolidation controls
    if 'consolidation_df' in st.session_state:
        df = st.session_state.consolidation_df

        st.divider()
        st.subheader("⚙️ Consolidation Settings")

        col1, col2 = st.columns(2)

        with col1:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Higher values = stricter matching. Lower values = more consolidations."
            )

        with col2:
            # Let user select which column contains the requirement text
            text_columns = [col for col in df.columns if pd.notna(col)]
            requirement_column = st.selectbox(
                "Requirement Text Column",
                options=text_columns,
                index=0 if 'Requirement (Clause)' not in text_columns else text_columns.index('Requirement (Clause)'),
                help="Select the column containing the requirement descriptions"
            )

        # Create two columns for different consolidation methods
        col1, col2 = st.columns(2)

        with col1:
            # Run AI consolidation button
            if st.button("🤖 Analyze with AI (Original)", type="secondary"):
                with st.spinner("🔬 Analyzing with AI..."):
                    try:
                        # Convert DataFrame to list of dicts
                        requirements = df.to_dict('records')

                        # Run AI analysis
                        consolidations = analyze_similarity_with_ai(
                            requirements,
                            st.session_state.anthropic_api_key,
                            similarity_threshold
                        )

                        # Store results
                        st.session_state.consolidations = consolidations
                        st.session_state.consolidation_method = 'ai'

                        st.success(f"✅ Found {len(consolidations)} consolidation opportunities!")

                    except Exception as e:
                        st.error(f"❌ Error during AI analysis: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        with col2:
            # Run improved consolidation button
            if st.button("⚡ Core + Deltas (Improved)", type="primary"):
                with st.spinner("🔧 Processing with Core + Deltas approach..."):
                    try:
                        # Use improved consolidator
                        consolidator = ImprovedConsolidator()
                        consolidator.MIN_SIMILARITY = similarity_threshold
                        
                        result = consolidator.consolidate_from_dataframe(df)
                        
                        # Convert result to format expected by display function
                        consolidations = []
                        groups_df = result['consolidation_groups']
                        
                        for idx, row in groups_df.iterrows():
                            consolidations.append({
                                'group_id': row['Group ID'],
                                'similarity_score': row['Similarity Score'],
                                'topic_keywords': [row['Topic Keywords']],
                                'reasoning': row['Reasoning'],
                                'suggested_consolidation': row['Suggested Consolidation'],
                                'original_requirements': row['Original Requirement Indices'],
                                'critical_differences': row.get('Critical Differences', [])
                            })
                        
                        # Store results
                        st.session_state.consolidations = consolidations
                        st.session_state.consolidation_stats = result['statistics']
                        st.session_state.consolidation_method = 'improved'

                        st.success(f"✅ Created {len(consolidations)} consolidated groups!")
                        
                        # Show statistics
                        stats = result['statistics']
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Original", stats['original_requirements'])
                        with col_b:
                            st.metric("Groups", stats['consolidated_groups'])
                        with col_c:
                            st.metric("Ungrouped", stats['ungrouped_count'])

                    except Exception as e:
                        st.error(f"❌ Error during consolidation: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # Display consolidation results
        if 'consolidations' in st.session_state and st.session_state.consolidations:
            st.divider()
            st.subheader("📊 Consolidation Suggestions")

            consolidations = st.session_state.consolidations

            for i, consolidation in enumerate(consolidations, 1):
                with st.expander(
                    f"**Group {i}** - Similarity: {consolidation['similarity_score']:.0%} - "
                    f"{', '.join(consolidation['topic_keywords'][:3])}"
                ):
                    # Show reasoning
                    st.markdown(f"**💡 Reasoning:** {consolidation['reasoning']}")

                    # Show original requirements
                    st.markdown("**📄 Original Requirements:**")
                    orig_indices = consolidation['original_requirements']
                    for idx, req_idx in enumerate(orig_indices, 1):
                        req = df.iloc[req_idx]
                        text = req.get(requirement_column, req.get('Description', ''))
                        standard = req.get('Standard/ Regulation', req.get('Standard/Reg', ''))
                        clause = req.get('Clause', req.get('Clause/Requirement', ''))

                        # Convert to string and handle NaN/empty values
                        text_str = str(text) if pd.notna(text) else '[Empty]'
                        standard_str = str(standard) if pd.notna(standard) else '[Unknown]'
                        clause_str = str(clause) if pd.notna(clause) else '[Unknown]'

                        # Truncate text safely
                        text_display = text_str[:200] + '...' if len(text_str) > 200 else text_str

                        st.markdown(f"""
                        {idx}. **{standard_str}** - Clause {clause_str}
                        _{text_display}_
                        """)

                    # Show suggested consolidation
                    st.markdown("**✨ Suggested Consolidation:**")
                    st.info(consolidation['suggested_consolidation'])

                    # Show critical differences if any
                    if consolidation.get('critical_differences'):
                        st.markdown("**⚠️ Critical Differences to Preserve:**")
                        for diff in consolidation['critical_differences']:
                            st.markdown(f"- {diff}")

                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Accept", key=f"accept_{i}"):
                            st.success("Consolidation accepted! (Implementation pending)")
                    with col2:
                        if st.button(f"❌ Reject", key=f"reject_{i}"):
                            st.info("Consolidation rejected")

            # Export consolidated results
            st.divider()
            if st.button("📥 Export Results"):
                # Create export DataFrame with consolidations
                export_data = []
                for cons in consolidations:
                    export_data.append({
                        'Group ID': cons['group_id'],
                        'Similarity Score': cons['similarity_score'],
                        'Topic Keywords': ', '.join(cons['topic_keywords']),
                        'Reasoning': cons['reasoning'],
                        'Suggested Consolidation': cons['suggested_consolidation'],
                        'Original Requirement Indices': ', '.join(map(str, cons['original_requirements']))
                    })

                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Consolidation Report",
                    data=csv,
                    file_name="consolidation_report.csv",
                    mime="text/csv",
                    type="primary"
                )


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

        # Keep only the 7 schema columns
        display_columns = [
            'Description',
            'Standard/Reg',
            'Clause/Requirement',
            'Requirement scope',
            'Formatting required?',
            'Required in Print?',
            'Comments'
        ]

        # Initialize edited data in session state if not exists
        if 'edited_data' not in st.session_state or st.session_state.get('current_job_id') != st.session_state.job_id:
            st.session_state.edited_data = df[display_columns].copy()
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

        # Editable DataFrame
        edited_df = st.data_editor(
            st.session_state.edited_data,
            use_container_width=True,
            hide_index=False,
            height=500,
            num_rows="dynamic",  # Allow adding/deleting rows
            key="data_editor"
        )

        # Update session state with edited data
        st.session_state.edited_data = edited_df

        st.caption(f"Total requirements: {len(edited_df)}")

        # Export button - one-click download using edited data
        st.divider()

        # Convert edited DataFrame to CSV
        csv_data = edited_df.to_csv(index=False)

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"{result['filename']}_requirements.csv",
            mime="text/csv",
            type="primary"
        )

    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")


if __name__ == "__main__":
    main()
