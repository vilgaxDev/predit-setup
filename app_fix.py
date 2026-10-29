"""
app_fix.py
==========
Backup of fixed app.py section to replace the problematic code.
"""

# This is the fixed version of lines 543-556 in app.py
# Replace the problematic section with this:

        if total_loaded > 0:
            st.sidebar.success(f"Successfully loaded {total_loaded} data sources")
            # Set default to first maternity sheet if available, otherwise first available
            maternity_keys = [k for k in st.session_state.all_data_sources.keys() if k.startswith('maternity_')]
            if maternity_keys:
                first_key = maternity_keys[0]
                st.session_state.current_df = st.session_state.all_data_sources[first_key]['data']
                st.session_state.data_source = f"integrated ({st.session_state.all_data_sources[first_key]['name']})"
            elif st.session_state.all_data_sources:
                first_key = list(st.session_state.all_data_sources.keys())[0]
                st.session_state.current_df = st.session_state.all_data_sources[first_key]['data']
                st.session_state.data_source = f"integrated ({st.session_state.all_data_sources[first_key]['name']})"
        else:
            st.sidebar.warning("No data files found. Please upload data below.")
