"""
Streamlit UI for E-Bike Standards Requirement Extractor.
"""
import streamlit as st
import requests
import pandas as pd
import time


# Configuration
API_BASE_URL = "http://localhost:8000"


def main():
    st.set_page_config(
        page_title="E-Bike Standards Extractor",
        page_icon="🚴",
        layout="wide"
    )

    st.title("🚴 E-Bike Standards Requirement Extractor")
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
        st.header("📁 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF containing e-bike standards or regulations"
        )

        standard_name = st.text_input(
            "Standard Name (optional)",
            placeholder="e.g., EN 15194, 16 CFR Part 1512",
            help="Name of the standard/regulation being analyzed"
        )

        custom_section_name = st.text_input(
            "Custom Section Name (optional)",
            placeholder="e.g., Instruction for use",
            help="Specific section name to look for in the document. Leave blank to use default patterns."
        )

        process_button = st.button("🔍 Process Document", type="primary", use_container_width=True)

    # Main content area
    if process_button and uploaded_file:
        process_document(uploaded_file, standard_name, custom_section_name)

    # Show existing results if available
    if 'job_id' in st.session_state:
        display_results(st.session_state.job_id)


def check_api_health():
    """Check if FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False


def process_document(uploaded_file, standard_name, custom_section_name):
    """Process uploaded document through the API."""
    with st.spinner("🔄 Extracting and analyzing document..."):
        try:
            # Prepare file and form data
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')
            }

            params = {}
            if standard_name:
                params['standard_name'] = standard_name
            if custom_section_name:
                params['custom_section_name'] = custom_section_name

            # Call upload API
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                params=params,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                st.session_state.job_id = result['job_id']

                # Show success message
                st.success("✅ Document processed successfully!")

                # Show extraction stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Extraction Method", result.get('extraction_method', 'unknown'))
                with col2:
                    st.metric("Confidence", result.get('extraction_confidence', 'unknown').upper())
                with col3:
                    st.metric("Total Detected", result['stats']['total_detected'])
                with col4:
                    st.metric("Classified Rows", result['stats']['classified_rows'])

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

        # Keep only the 8 schema columns
        display_columns = [
            'Description',
            'Standard/Reg',
            'Clause/Requirement',
            'Must Be Included with product?',
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
        st.info("💡 **Tip**: Click any cell to edit. Select rows to delete using checkboxes on the left.")

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

        # Consolidation suggestions (Phase 3 - placeholder)
        st.divider()
        st.subheader("🔗 Consolidation Suggestions")
        consolidations = result.get('consolidations', [])

        if consolidations:
            for i, group in enumerate(consolidations, 1):
                with st.expander(f"Group {i}: {group.get('representative_text', '')[:100]}..."):
                    st.write(f"**Reason:** {group.get('reason', 'N/A')}")
                    st.write(f"**Scope:** {group.get('scope', 'N/A')}")
                    st.write(f"**Members:** {len(group.get('members', []))}")
                    st.json(group.get('members', []))
        else:
            st.info("Consolidation feature coming in Phase 3")

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
