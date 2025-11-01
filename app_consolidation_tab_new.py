# REPLACE render_consolidation_tab() function in app.py with this:

def render_consolidation_tab():
    """Tab 2: Smart AI Consolidation"""
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

            st.success(f"✅ Loaded {len(df)} rows")

            # Show preview
            with st.expander(f"📋 Data Preview (first 10 rows)"):
                st.dataframe(df.head(10))

            # Store in session state
            st.session_state.consolidation_df = df

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

    # If data is loaded, show consolidation controls
    if 'consolidation_df' in st.session_state:
        df = st.session_state.consolidation_df

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
        
        # Run consolidation button
        if st.button("🧠 Analyze with Smart AI", type="primary", use_container_width=True):
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
            
            # Display each group
            for group in result['groups']:
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
                
                with st.expander(
                    f"{status_emoji} **Group {group.group_id + 1}: {group.topic}** "
                    f"({len(group.requirement_indices)} requirements) - "
                    f"{group.consolidation_potential:.0%} match"
                ):
                    # Regulatory Intent
                    st.markdown("### 🎯 Regulatory Intent")
                    st.info(group.regulatory_intent)
                    
                    # Core Requirement
                    st.markdown("### 📌 Core Requirement (Consolidated)")
                    st.success(group.core_requirement)
                    
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
                    
                    # Show original requirements
                    with st.expander("📄 View Original Requirements"):
                        for idx in group.requirement_indices:
                            if idx < len(df):
                                req_row = df.iloc[idx]
                                req_text = req_row.get('Requirement (Clause)', req_row.get('Description', ''))
                                standard = req_row.get('Standard/ Regulation', req_row.get('Standard/Reg', ''))
                                clause = req_row.get('Clause ID', req_row.get('Clause/Requirement', ''))
                                
                                st.markdown(f"**{standard}** (Clause {clause})")
                                st.caption(str(req_text)[:300] + "..." if len(str(req_text)) > 300 else str(req_text))
                                st.divider()
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"✅ Accept", key=f"accept_smart_{group.group_id}"):
                            st.success("Group accepted!")
                    with col2:
                        if st.button(f"✏️ Edit", key=f"edit_smart_{group.group_id}"):
                            st.info("Edit functionality coming soon")
                    with col3:
                        if st.button(f"❌ Reject", key=f"reject_smart_{group.group_id}"):
                            st.warning("Group rejected")
            
            # Export results
            st.divider()
            if st.button("📥 Export Consolidation Report", type="primary"):
                # Create export DataFrame
                export_data = []
                for group in result['groups']:
                    export_data.append({
                        'Group ID': group.group_id + 1,
                        'Topic': group.topic,
                        'Regulatory Intent': group.regulatory_intent,
                        'Core Requirement': group.core_requirement,
                        'Applies To Standards': ', '.join(group.applies_to_standards),
                        'Critical Differences': '; '.join(group.critical_differences),
                        'Consolidation Potential': f"{group.consolidation_potential:.0%}",
                        'Requirement Count': len(group.requirement_indices),
                        'Original Indices': ', '.join(map(str, group.requirement_indices))
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv,
                    file_name="smart_consolidation_report.csv",
                    mime="text/csv"
                )
