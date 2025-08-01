import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import base64
import plotly.io as pio
from matplotlib import pyplot as plt
import networkx as nx
from plotly.colors import sample_colorscale
import math
import control as ct
import datetime
import csv
import os


def save_feedback_to_csv(rating, category, feedback_text):
    """Save user feedback to a CSV file."""
    try:
        # Define the CSV file path
        csv_file = "feedback_log.csv"
        
        # Prepare the feedback data
        feedback_data = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'rating': rating,
            'category': category,
            'feedback': feedback_text.replace('\n', ' ').replace('\r', ' ')  # Clean newlines
        }
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.isfile(csv_file)
        
        # Write to CSV file
        with open(csv_file, 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['timestamp', 'rating', 'category', 'feedback']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the feedback data
            writer.writerow(feedback_data)
        
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return False


def lotka_volterra(mu_1, mu_2, alpha_11, alpha_12, alpha_21, alpha_22):
    def lv_ode(t, pops):
        x1, x2 = pops
        dx1 = x1 * (mu_1 + alpha_11 * x1 + alpha_12 * x2)
        dx2 = x2 * (mu_2 + alpha_22 * x2 + alpha_21 * x1)
        return [dx1, dx2]
    return lv_ode

def lv_sim(mu_1, mu_2, alpha_11, alpha_12, alpha_21, alpha_22, x1_0, x2_0, time_span):
    lv_ode = lotka_volterra(mu_1, mu_2, alpha_11, alpha_12, alpha_21, alpha_22)
    sol = solve_ivp(
        lv_ode,
        (time_span[0], time_span[-1]),
        [x1_0, x2_0],
        t_eval=time_span,
        method='RK45'
    )
    if not sol.success or np.any(np.isnan(sol.y)):
        return np.full_like(time_span, np.inf), np.full_like(time_span, np.inf), np.full_like(time_span, np.inf)
    sim_x1 = sol.y[0]
    sim_x2 = sol.y[1]
    sim_sum = sim_x1 + sim_x2
    return sim_x1, sim_x2, sim_sum

def mse_obj(params, mu_1, mu_2, x1_0, x2_0, time_span, observed_sum):
    a11, a12, a21, a22 = params
    _, _, sim_sum = lv_sim(mu_1, mu_2, a11, a12, a21, a22, x1_0, x2_0, time_span)
    return np.mean((sim_sum - observed_sum) ** 2)


st.set_page_config(
    page_title="LV-Sim (Lotka-Volterra Simulator)",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .section-header {
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #1976D2;
    }
    .info-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def read_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, header=None)
        elif file.name.endswith('.xls'):
            try:
                return pd.read_excel(file, engine='xlrd', header=None)
            except ImportError:
                st.error("📋 **Excel .xls files require the 'xlrd' library.**")
                st.info("Please install it using: `pip install xlrd>=2.0.1` or upload a CSV/.xlsx file instead.")
                return None
        elif file.name.endswith('.xlsx'):
            try:
                return pd.read_excel(file, engine='openpyxl', header=None)
            except ImportError:
                st.error("📋 **Excel .xlsx files require the 'openpyxl' library.**")
                st.info("Please install it using: `pip install openpyxl` or upload a CSV file instead.")
                return None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel (.xls/.xlsx) file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        if "xlrd" in str(e).lower():
            st.info("💡 **Tip**: Try converting your Excel file to CSV format, which doesn't require additional libraries.")
        return None

def read_file_with_headers(file, has_headers=True):
    """Read file with option to treat first row as headers"""
    try:
        header = 0 if has_headers else None
        if file.name.endswith('.csv'):
            return pd.read_csv(file, header=header)
        elif file.name.endswith('.xls'):
            try:
                return pd.read_excel(file, engine='xlrd', header=header)
            except ImportError:
                st.error("📋 **Excel .xls files require the 'xlrd' library.**")
                st.info("Please install it using: `pip install xlrd>=2.0.1` or upload a CSV/.xlsx file instead.")
                return None
        elif file.name.endswith('.xlsx'):
            try:
                return pd.read_excel(file, engine='openpyxl', header=header)
            except ImportError:
                st.error("📋 **Excel .xlsx files require the 'openpyxl' library.**")
                st.info("Please install it using: `pip install openpyxl` or upload a CSV file instead.")
                return None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel (.xls/.xlsx) file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        if "xlrd" in str(e).lower():
            st.info("💡 **Tip**: Try converting your Excel file to CSV format, which doesn't require additional libraries.")
        return None

def ode_model(t, x0, mu):
    """
    Exponential growth model: dx/dt = mu * x
    Solution: x(t) = x0 * exp(mu * t)
    """
    # Use analytical solution instead of numerical integration for better stability
    return x0 * np.exp(mu * (t - t[0]))

def download_csv(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename} CSV file</a>'
    return href

def get_svg_download_link(fig, filename):
    svg_bytes = pio.to_image(fig, format="svg")
    b64 = base64.b64encode(svg_bytes).decode()
    href = f'<a href="data:image/svg+xml;base64,{b64}" download="{filename}.svg">Download {filename} as SVG</a>'
    return href

st.markdown('<h1 class="main-header">LV-Sim (Lotka-Volterra Simulator)</h1>', unsafe_allow_html=True)
main_tabs = st.tabs(["Data Upload", "Data Analysis", "Model Fitting", "About"])

# ========== Data Upload Tab ==========
with main_tabs[0]:
    st.markdown('<h2 class="section-header">Upload and Process Data</h2>', unsafe_allow_html=True)
    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        **New Flexible Data Upload Process:**
        
        1. **Select number of replicate files** - Choose how many replicate files you have
        2. **Upload each file** (CSV or Excel format) - See your raw data displayed under each upload  
        3. **Configure species count** - Specify how many species are in your experiment
        4. **Map your columns** - Tell the app which columns in your file correspond to:
           - Time data
           - Individual species measurements (e.g., Species 1, Species 2, Species 3...)  
           - **All pairwise co-culture combinations** (e.g., for 3 species: Species 1+2, Species 1+3, Species 2+3)
           - Background measurements (optional)
        5. **Process data** - The app will average your replicates and prepare data for analysis
        
        **📊 Expected Data Structure Examples:**
        - **2 species**: Time, Species 1, Species 2, Species 1+2, Background
        - **3 species**: Time, Species 1, Species 2, Species 3, Species 1+2, Species 1+3, Species 2+3, Background  
        - **4 species**: Time, Species 1-4, Species 1+2, Species 1+3, Species 1+4, Species 2+3, Species 2+4, Species 3+4, Background
        
        **Benefits of the new system:**
        - ✅ **No strict column order required** - your columns can be in any order
        - ✅ **Column headers supported** - the app can read and use your column names
        - ✅ **Smart defaults** - the app suggests likely column mappings
        - ✅ **Clear validation** - see exactly how your data will be processed before confirming
        - ✅ **Handles any number of species** - automatically generates all pairwise combinations
        
        **📁 File Format Support:**
        - **CSV files**: Fully supported (recommended format)
        - **Excel files (.xlsx)**: Supported (requires `openpyxl` library)
        - **Excel files (.xls)**: Supported (requires `xlrd` library)
        
        💡 **Tip**: If you encounter library errors, convert your Excel files to CSV format for the best compatibility.
        """)
    col1, col2 = st.columns([1, 2])
    with col1:
        num_replicates = st.number_input("Number of replicate files", min_value=1, max_value=10, step=1, key="num_replicates")
    uploaded_files = []
    dfs = []
    file_cols = st.columns(min(3, int(num_replicates)))
    for i in range(int(num_replicates)):
        with file_cols[i % len(file_cols)]:
            f = st.file_uploader(f"Replicate file {i+1}", type=['csv', 'xls', 'xlsx'], key=f"file_{i}")
            if f is not None:
                uploaded_files.append(f)
                df_raw = read_file(f)
                st.success(f"✅ {f.name}")
                st.write("⬇️ **Raw data preview:**")
                st.dataframe(df_raw, use_container_width=True)
                dfs.append(df_raw)
    file_progress = len(uploaded_files) / int(num_replicates) if int(num_replicates) > 0 else 0
    if file_progress < 1:
        st.progress(file_progress, text=f"Uploaded {len(uploaded_files)} of {int(num_replicates)} files")
    elif int(num_replicates) > 0:
        st.success(f"All {int(num_replicates)} files uploaded!")
    if len(uploaded_files) == int(num_replicates) and int(num_replicates) > 0:
        # Step 1: Get number of species
        st.subheader("Step 1: Configure Species Count")
        if 'confirmed_species_count_val' not in st.session_state:
            st.session_state.confirmed_species_count_val = 1
        
        species_count_from_form = st.number_input(
            "Number of species in your experiment", 
            min_value=1, 
            max_value=20, 
            step=1,
            value=st.session_state.confirmed_species_count_val,
            key="species_count_input_widget",
            help="This determines how many individual species and pairwise combinations we expect in your data"
        )
        
        # Show expected data structure
        if species_count_from_form > 0:
            with st.expander("📊 Expected Data Structure for Your Experiment", expanded=True):
                # Calculate expected columns
                individual_species = [f"Species {i+1}" for i in range(species_count_from_form)]
                pairwise_combinations = []
                for i in range(species_count_from_form):
                    for j in range(i+1, species_count_from_form):
                        pairwise_combinations.append(f"Species {i+1}+{j+1}")
                
                total_combinations = len(pairwise_combinations)
                total_expected_cols = 1 + species_count_from_form + total_combinations + 1  # Time + Species + Pairwise + Background
                
                st.markdown(f"""
                **For {species_count_from_form} species, your data should contain:**
                
                **Required columns:**
                - **1 Time column**: Time points for your experiment
                - **{species_count_from_form} Individual species columns**: {', '.join(individual_species)}
                - **{total_combinations} Pairwise combination columns**: {', '.join(pairwise_combinations)}
                - **1 Background column (optional)**: Background/blank measurements
                
                **Total expected columns: {total_expected_cols}** (including optional background)
                **Total minimum columns: {total_expected_cols-1}** (excluding optional background)
                """)
                
                if total_combinations > 10:
                    st.warning(f"⚠️ With {species_count_from_form} species, you'll need {total_combinations} pairwise combinations. This grows quickly: n species requires n×(n-1)/2 pairwise combinations.")
        
        # Step 2: Column mapping interface
        if species_count_from_form > 0:
            st.subheader("Step 2: Map Your Data Columns")
            
            # Get column headers from the first uploaded file
            sample_df = dfs[0] if dfs else None
            if sample_df is not None:
                # Try to detect if first row contains headers
                first_row = sample_df.iloc[0].astype(str)
                is_likely_header = any(
                    any(char.isalpha() for char in str(val)) for val in first_row
                )
                
                with st.expander("🔍 Detect Column Headers", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        has_headers = st.checkbox(
                            "My files have column headers in the first row", 
                            value=is_likely_header,
                            help="Check this if your first row contains column names instead of data"
                        )
                    with col2:
                        if has_headers:
                            st.info("First row will be treated as column headers")
                        else:
                            st.info("First row will be treated as data")
                
                # Get available columns
                if has_headers:
                    try:
                        # Re-read files with headers
                        if sample_df.name.endswith('.csv') if hasattr(sample_df, 'name') else uploaded_files[0].name.endswith('.csv'):
                            temp_df = pd.read_csv(uploaded_files[0], header=0)
                        elif uploaded_files[0].name.endswith('.xls'):
                            try:
                                temp_df = pd.read_excel(uploaded_files[0], engine='xlrd', header=0)
                            except ImportError:
                                st.error("📋 **Excel .xls files require the 'xlrd' library.**")
                                st.info("Please install it using: `pip install xlrd>=2.0.1` or upload a CSV/.xlsx file instead.")
                                available_columns = [f"Column {i}" for i in range(sample_df.shape[1])]
                                temp_df = None
                        else:
                            try:
                                temp_df = pd.read_excel(uploaded_files[0], engine='openpyxl', header=0)
                            except ImportError:
                                st.error("📋 **Excel .xlsx files require the 'openpyxl' library.**")
                                st.info("Please install it using: `pip install openpyxl` or upload a CSV file instead.")
                                available_columns = [f"Column {i}" for i in range(sample_df.shape[1])]
                                temp_df = None
                        
                        if temp_df is not None:
                            available_columns = list(temp_df.columns)
                    except Exception as e:
                        st.warning(f"Could not read column headers: {str(e)}")
                        available_columns = [f"Column {i}" for i in range(sample_df.shape[1])]
                else:
                    available_columns = [f"Column {i}" for i in range(sample_df.shape[1])]
                
                st.info(f"Your file has {len(available_columns)} columns available for mapping.")
                
                # Generate expected data types based on species count
                species_labels = [f"Species {i+1}" for i in range(species_count_from_form)]
                pairwise_labels = []
                for i in range(species_count_from_form):
                    for j in range(i+1, species_count_from_form):
                        pairwise_labels.append(f"Species {i+1} + Species {j+1}")
                
                required_columns = ["Time"] + species_labels + pairwise_labels + ["Background (optional)"]
                
                with st.form(key='column_mapping_form'):
                    st.markdown("**Map each required data type to a column from your file:**")
                    
                    column_mapping = {}
                    
                    # Time column (always first)
                    st.markdown("### 🕒 Time Data")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Look for time-related column names
                        time_default_idx = 0
                        time_keywords = ['time', 'hour', 'day', 'minute', 't']
                        for i, col in enumerate(available_columns):
                            if any(keyword in str(col).lower() for keyword in time_keywords):
                                time_default_idx = i
                                break
                        
                        time_column = st.selectbox(
                            "**Time** *",
                            options=available_columns,
                            index=time_default_idx,
                            key="mapping_Time",
                            help="Required: Select the column containing time points"
                        )
                        column_mapping["Time"] = time_column
                    
                    # Individual species columns
                    st.markdown("### 🧬 Individual Species")
                    species_cols = st.columns(min(3, len(species_labels)))
                    for idx, species_label in enumerate(species_labels):
                        with species_cols[idx % len(species_cols)]:
                            default_idx = min(idx + 1, len(available_columns) - 1)  # Skip time column
                            selected = st.selectbox(
                                f"**{species_label}** *",
                                options=available_columns,
                                index=default_idx,
                                key=f"mapping_{species_label}",
                                help=f"Required: Select column for {species_label} monoculture data"
                            )
                            column_mapping[species_label] = selected
                    
                    # Pairwise combination columns
                    if pairwise_labels:
                        st.markdown("### 🤝 Pairwise Co-cultures")
                        pairwise_cols = st.columns(min(3, len(pairwise_labels)))
                        for idx, pairwise_label in enumerate(pairwise_labels):
                            with pairwise_cols[idx % len(pairwise_cols)]:
                                default_idx = min(idx + 1 + len(species_labels), len(available_columns) - 1)
                                selected = st.selectbox(
                                    f"**{pairwise_label}** *",
                                    options=available_columns,
                                    index=default_idx,
                                    key=f"mapping_{pairwise_label}",
                                    help=f"Required: Select column for {pairwise_label} co-culture data"
                                )
                                column_mapping[pairwise_label] = selected
                    
                    # Background column (optional)
                    st.markdown("### 🔬 Background Control")
                    col1, col2 = st.columns(2)
                    with col1:
                        background_options = ["None (skip this column)"] + available_columns
                        background_selected = st.selectbox(
                            "**Background (optional)**",
                            options=background_options,
                            index=0,
                            key="mapping_Background (optional)",
                            help="Optional: Select column for background/blank measurements"
                        )
                        if background_selected != "None (skip this column)":
                            column_mapping["Background (optional)"] = background_selected
                    
                    # Additional settings
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        subtract_bg_form = st.checkbox(
                            "Subtract background from data?", 
                            value=False,
                            help="If enabled, background values will be subtracted from all other measurements"
                        )
                    with col2:
                        st.markdown("*Required fields")
                    
                    # Validation and preview
                    with st.expander("📋 Mapping Preview", expanded=True):
                        st.markdown("**Your column mapping:**")
                        for req_col, mapped_col in column_mapping.items():
                            st.write(f"• **{req_col}** → `{mapped_col}`")
                        
                        # Check for duplicate mappings
                        mapped_values = [v for v in column_mapping.values() if v != "None (skip this column)"]
                        duplicates = set([x for x in mapped_values if mapped_values.count(x) > 1])
                        if duplicates:
                            st.error(f"⚠️ Duplicate mappings detected: {', '.join(duplicates)}. Each column can only be mapped once.")
                    
                    process_data_button = st.form_submit_button("Process Data with Column Mapping", use_container_width=True)
                    
                    if process_data_button:
                        # Validate mapping
                        mapped_values = [v for v in column_mapping.values() if v != "None (skip this column)"]
                        duplicates = set([x for x in mapped_values if mapped_values.count(x) > 1])
                        if duplicates:
                            st.error(f"Cannot process: Duplicate column mappings detected: {', '.join(duplicates)}")
                        else:
                            with st.spinner("Processing data with your column mapping..."):
                                try:
                                    st.session_state.confirmed_species_count_val = species_count_from_form
                                    
                                    # Re-read all files with proper headers
                                    processed_dfs = []
                                    for file_obj in uploaded_files:
                                        df = None
                                        if has_headers:
                                            if file_obj.name.endswith('.csv'):
                                                df = pd.read_csv(file_obj, header=0)
                                            elif file_obj.name.endswith('.xls'):
                                                try:
                                                    df = pd.read_excel(file_obj, engine='xlrd', header=0)
                                                except ImportError:
                                                    st.error("📋 **Excel .xls files require the 'xlrd' library.**")
                                                    st.info("Please install it using: `pip install xlrd>=2.0.1` or upload a CSV/.xlsx file instead.")
                                                    st.stop()
                                            else:
                                                try:
                                                    df = pd.read_excel(file_obj, engine='openpyxl', header=0)
                                                except ImportError:
                                                    st.error("📋 **Excel .xlsx files require the 'openpyxl' library.**")
                                                    st.info("Please install it using: `pip install openpyxl` or upload a CSV file instead.")
                                                    st.stop()
                                        else:
                                            # Use the original dataframes (no headers)
                                            df = read_file(file_obj)
                                            if df is None:
                                                st.error(f"Failed to read file {file_obj.name}")
                                                st.stop()
                                            df.columns = available_columns
                                        
                                        if df is None:
                                            st.error(f"Failed to process file {file_obj.name}")
                                            st.stop()
                                        
                                        # Extract and reorder columns based on mapping
                                        mapped_df_data = {}
                                        for req_col, file_col in column_mapping.items():
                                            if file_col in df.columns:
                                                mapped_df_data[req_col] = df[file_col]
                                        
                                        mapped_df = pd.DataFrame(mapped_df_data)
                                        processed_dfs.append(mapped_df)
                                    
                                    # Verify all files have the same structure
                                    if not all(df.shape == processed_dfs[0].shape for df in processed_dfs):
                                        st.error("Files have different structures after mapping. Please check your files.")
                                    else:
                                        # Average across replicates
                                        time_data = processed_dfs[0]["Time"].values
                                        
                                        # Check time consistency across files
                                        for i, df in enumerate(processed_dfs[1:], 1):
                                            if not np.allclose(df["Time"].values, time_data, rtol=1e-3):
                                                st.warning(f"Time values in file {i+1} don't match file 1. Using file 1 as reference.")
                                        
                                        # Average the data columns
                                        data_columns = [col for col in column_mapping.keys() if col != "Time"]
                                        averaged_data = {}
                                        averaged_data["time"] = time_data
                                        
                                        # Create standardized column names
                                        species_cols = []
                                        pairwise_cols = []
                                        
                                        # Process species columns
                                        for i in range(species_count_from_form):
                                            species_label = f"Species {i+1}"
                                            if species_label in processed_dfs[0].columns:
                                                values = np.array([df[species_label].values for df in processed_dfs])
                                                new_name = f"x{i+1}"
                                                averaged_data[new_name] = np.mean(values, axis=0)
                                                species_cols.append(new_name)
                                        
                                        # Process pairwise columns
                                        for i in range(species_count_from_form):
                                            for j in range(i+1, species_count_from_form):
                                                pairwise_label = f"Species {i+1} + Species {j+1}"
                                                if pairwise_label in processed_dfs[0].columns:
                                                    values = np.array([df[pairwise_label].values for df in processed_dfs])
                                                    new_name = f"x{i+1}+x{j+1}"
                                                    averaged_data[new_name] = np.mean(values, axis=0)
                                                    pairwise_cols.append(new_name)
                                        
                                        # Process background if present
                                        if "Background (optional)" in processed_dfs[0].columns:
                                            values = np.array([df["Background (optional)"].values for df in processed_dfs])
                                            averaged_data["background_avg"] = np.mean(values, axis=0)
                                        else:
                                            # If no background, add a column of zeros
                                            averaged_data["background_avg"] = np.zeros_like(time_data)
                                        
                                        # Create final dataframe
                                        final_columns = ["time"] + species_cols + pairwise_cols + ["background_avg"]
                                        df_avg = pd.DataFrame({col: averaged_data[col] for col in final_columns})
                                        
                                        # Store in session state
                                        st.session_state.df_avg = df_avg
                                        st.session_state.species_count = species_count_from_form
                                        st.session_state.subtract_bg = subtract_bg_form
                                        st.session_state.species_cols = species_cols
                                        st.session_state.pairwise_cols = pairwise_cols
                                        st.session_state.column_mapping = column_mapping
                                        
                                        st.success("✅ Data processed successfully with custom column mapping! Go to the Data Analysis section.")
                                        st.balloons()
                                        
                                        # Show preview of processed data
                                        with st.expander("📊 Processed Data Preview", expanded=True):
                                            st.write("First 5 rows of your processed data:")
                                            st.dataframe(df_avg.head(), use_container_width=True)
                                        
                                except Exception as e:
                                    st.error(f"Error processing data: {str(e)}")
                                    st.error("Please check your column mapping and file format.")
            
            else:
                st.warning("No data files detected. Please upload files first.")

# ========== Data Analysis Tab ==========
with main_tabs[1]:
    if 'df_avg' in st.session_state:
        st.markdown('<h2 class="section-header">Data Analysis</h2>', unsafe_allow_html=True)
        df_avg = st.session_state.df_avg
        species_count = st.session_state.species_count
        subtract_bg = st.session_state.subtract_bg
        species_cols = st.session_state.species_cols
        pairwise_cols = st.session_state.pairwise_cols
        data_tabs = st.tabs(["Data Table", "Individual Species", "Pairwise Co-cultures"])
        with data_tabs[0]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Averaged data from all replicates")
            with col2:
                st.markdown(download_csv(df_avg, "averaged_data"), unsafe_allow_html=True)
            if subtract_bg:
                data_no_time = df_avg.iloc[:, 1:-1].subtract(df_avg["background_avg"], axis=0)
                st.info("Background subtraction was applied to this data.")
            else:
                data_no_time = df_avg.iloc[:, 1:-1]
            col1, col2 = st.columns([1, 1])
            with col1:
                view_option = st.radio("View data:", ["Raw data", "Processed data (after background subtraction)"])
            with col2:
                show_rows = st.slider("Number of rows to display", 5, min(100, len(df_avg)), 10)
            if view_option == "Raw data":
                st.dataframe(df_avg.head(show_rows), use_container_width=True)
            else:
                display_df = pd.concat([df_avg[["time"]], data_no_time], axis=1)
                st.dataframe(display_df.head(show_rows), use_container_width=True)
        with data_tabs[1]:
            st.subheader("Individual Species Growth Curves")
            col1, col2 = st.columns([4, 1])
            with col2:
                y_scale = st.radio("Y-axis scale:", ["Linear", "Logarithmic"])
            fig = go.Figure()
            colors = px.colors.qualitative.Bold
            for idx, col in enumerate(species_cols):
                fig.add_trace(go.Scatter(
                    x=df_avg["time"],
                    y=data_no_time[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Optical Density (OD)",
                template="plotly_white",
                height=500,
                showlegend=True,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            if y_scale == "Logarithmic":
                fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(download_csv(pd.concat([df_avg[["time"]], data_no_time[species_cols]], axis=1), "individual_species_data"), unsafe_allow_html=True)
            st.markdown(get_svg_download_link(fig, "individual_species_growth_curves"), unsafe_allow_html=True)
        with data_tabs[2]:
            st.subheader("Pairwise Co-culture Growth Curves")
            col1, col2 = st.columns([4, 1])
            with col2:
                y_scale_pair = st.radio("Y-axis scale:", ["Linear", "Logarithmic"], key="pair_scale")
            fig_pair = go.Figure()
            for idx, col in enumerate(pairwise_cols):
                fig_pair.add_trace(go.Scatter(
                    x=df_avg["time"],
                    y=data_no_time[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            fig_pair.update_layout(
                xaxis_title="Time",
                yaxis_title="Optical Density (OD)",
                template="plotly_white",
                height=500,
                showlegend=True,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            if y_scale_pair == "Logarithmic":
                fig_pair.update_yaxes(type="log")
            st.plotly_chart(fig_pair, use_container_width=True)
            st.markdown(download_csv(pd.concat([df_avg[["time"]], data_no_time[pairwise_cols]], axis=1), "pairwise_coculture_data"), unsafe_allow_html=True)
            st.markdown(get_svg_download_link(fig_pair, "pairwise_coculture_growth_curves"), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process data first!")
        st.markdown("Go to the **Data Upload** section to get started.")

# ========== Model Fitting Tab ==========
with main_tabs[2]:
    if 'df_avg' in st.session_state:
        st.markdown('<h2 class="section-header">Model Fitting</h2>', unsafe_allow_html=True)
        with st.expander("ℹ️ About the ODE Model", expanded=False):
            st.markdown("""
            The exponential growth model used is:
            """)
            st.latex(r"\frac{dx}{dt} = \mu x")
            st.markdown("""
            where:
            - $\\mu$ is the growth rate
            - $x_0$ is the initial population
            """)
        df_avg = st.session_state.df_avg
        species_count = st.session_state.species_count
        subtract_bg = st.session_state.subtract_bg
        species_cols = st.session_state.species_cols
        if subtract_bg:
            data_no_time = df_avg.iloc[:, 1:-1].subtract(df_avg["background_avg"], axis=0)
        else:
            data_no_time = df_avg.iloc[:, 1:-1]
        time_data_full = df_avg["time"].values
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Configure Fitting Parameters")
        col1, col2 = st.columns(2)
        with col1:
            time_min, time_max = st.slider(
                "Select time range for fitting",
                min_value=float(np.min(time_data_full)),
                max_value=float(np.max(time_data_full)),
                value=(float(np.min(time_data_full)), float(np.max(time_data_full))),
                step=0.01
            )
        with col2:
            log_scale = st.checkbox("Use logarithmic Y-axis for plots", value=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate fit indices properly
        fit_indices = np.where((time_data_full >= time_min) & (time_data_full <= time_max))[0]
        time_data = time_data_full[fit_indices]
        
        # Debug information (remove after fixing if needed)
        st.write(f"Debug: Full time data length: {len(time_data_full)}")
        st.write(f"Debug: Selected time data length: {len(time_data)}")
        st.write(f"Debug: Fit indices length: {len(fit_indices)}")
        
        with st.spinner("Running model fitting..."):
            fit_results = []
            for species in species_cols:
                # Ensure both arrays use the same indices
                y_full = data_no_time[species].values
                
                # Critical fix: make sure indices are within bounds
                if len(fit_indices) > len(y_full):
                    st.error(f"Time selection extends beyond data range for {species}")
                    continue
                    
                y_data = y_full[fit_indices]
                
                # Verify lengths match
                if len(time_data) != len(y_data):
                    st.error(f"Length mismatch for {species}: time_data={len(time_data)}, y_data={len(y_data)}")
                    st.write(f"Time range: {time_min} to {time_max}")
                    st.write(f"Available data range: {time_data_full.min()} to {time_data_full.max()}")
                    continue
                
                try:
                    # Additional debugging for curve_fit
                    st.write(f"Debug before curve_fit: time_data length = {len(time_data)}, y_data length = {len(y_data)}")
                    st.write(f"Debug: time_data min/max = {time_data.min():.3f}/{time_data.max():.3f}")
                    st.write(f"Debug: y_data min/max = {y_data.min():.3f}/{y_data.max():.3f}")
                    
                    # Test the ode_model function before curve_fit
                    try:
                        test_result = ode_model(time_data, y_data[0], 0.1)
                        st.write(f"Debug: ode_model test successful, result length = {len(test_result)}")
                    except Exception as ode_error:
                        st.error(f"Debug: ode_model test failed: {ode_error}")
                        continue
                    
                    popt, pcov = curve_fit(ode_model, time_data, y_data, 
                                          p0=[y_data[0], 0.1],
                                          bounds=([1e-6, 0], [np.inf, np.inf]))
                    x0_fit, mu_fit = popt
                    residuals = y_data - ode_model(time_data, x0_fit, mu_fit)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_data - np.mean(y_data))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    perr = np.sqrt(np.diag(pcov))
                    x0_err, mu_err = perr

                    # Confidence Intervals (95%)
                    x0_CI_low = x0_fit - 1.96 * x0_err
                    x0_CI_high = x0_fit + 1.96 * x0_err
                    mu_CI_low = mu_fit - 1.96 * mu_err
                    mu_CI_high = mu_fit + 1.96 * mu_err

                    fit_results.append({
                        "Species": species, 
                        "Initial Value (x₀)": f"{x0_fit:.4f} ± {x0_err:.4f}",
                        "Growth Rate (μ)": f"{mu_fit:.4f} ± {mu_err:.4f}",
                        "R²": f"{r_squared:.4f}",
                        "x0_val": x0_fit,
                        "mu_val": mu_fit,
                        "x0_CI_low": x0_CI_low,
                        "x0_CI_high": x0_CI_high,
                        "mu_CI_low": mu_CI_low,
                        "mu_CI_high": mu_CI_high
                    })
                except Exception as e:
                    fit_results.append({
                        "Species": species, 
                        "Initial Value (x₀)": "Failed", 
                        "Growth Rate (μ)": "Failed", 
                        "R²": "N/A",
                        "x0_val": np.nan,
                        "mu_val": np.nan,
                        "x0_CI_low": np.nan,
                        "x0_CI_high": np.nan,
                        "mu_CI_low": np.nan,
                        "mu_CI_high": np.nan
                    })
                    st.error(f"Fit failed for {species}: {str(e)}")
                    st.write(f"Debug: time_data shape: {time_data.shape}")
                    st.write(f"Debug: y_data shape: {y_data.shape}")
                    st.write(f"Debug: fit_indices: {fit_indices[:5]}...{fit_indices[-5:] if len(fit_indices) > 5 else ''}")
            
            df_fit = pd.DataFrame(fit_results)
            st.session_state.df_fit = df_fit        

        st.subheader("Model Fitting Results")
        fit_tabs = st.tabs(["Parameter Table", "Fitted Curves", "Growth Rate Comparison", "Lotka-Volterra Pairwise", "LV Sum-Based Fit", "LV Plots"])
        with fit_tabs[0]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("ODE Model Parameters with 95% Confidence Intervals")
            with col2:
                st.markdown(download_csv(
                    df_fit[["Species", "Initial Value (x₀)", "Growth Rate (μ)", "R²", "x0_CI_low", "x0_CI_high", "mu_CI_low", "mu_CI_high"]], 
                    "fit_parameters_with_CI"), unsafe_allow_html=True)
            st.dataframe(df_fit[[
                "Species", 
                "Initial Value (x₀)", "x0_CI_low", "x0_CI_high",
                "Growth Rate (μ)", "mu_CI_low", "mu_CI_high",
                "R²"
            ]], use_container_width=True)

        with fit_tabs[1]:
            fig_fit = go.Figure()
            colors = px.colors.qualitative.Bold
            for idx, species in enumerate(species_cols):
                x0_fit = df_fit.loc[idx, "x0_val"]
                mu_fit = df_fit.loc[idx, "mu_val"]
                mu_CI_low = df_fit.loc[idx, "mu_CI_low"]
                mu_CI_high = df_fit.loc[idx, "mu_CI_high"]

                # Data curve
                fig_fit.add_trace(go.Scatter(
                    x=time_data_full,
                    y=data_no_time[species],
                    mode='lines',
                    name=f"{species} data",
                    line=dict(color=colors[idx % len(colors)]), opacity=0.7
                ))

                # Fit curve with CI bands for individual species exponential fits
                if not np.isnan(x0_fit) and not np.isnan(mu_fit):
                    fitted_curve = ode_model(time_data, x0_fit, mu_fit)
                    color_hex = colors[idx % len(colors)]
                    fig_fit.add_trace(go.Scatter(
                        x=time_data,
                        y=fitted_curve,
                        mode='lines',
                        name=f"{species} fit",
                        line=dict(color=color_hex, dash='dash', width=2)
                    ))
                    
                    # Add confidence intervals if available
                    if not np.isnan(mu_CI_low) and not np.isnan(mu_CI_high):
                        # Upper confidence bound
                        fitted_curve_upper = ode_model(time_data, x0_fit, mu_CI_high)
                        # Lower confidence bound  
                        fitted_curve_lower = ode_model(time_data, x0_fit, mu_CI_low)
                        
                        # Convert color to rgba for CI bands
                        if color_hex.startswith('#'):
                            r = int(color_hex[1:3], 16)
                            g = int(color_hex[3:5], 16)
                            b = int(color_hex[5:7], 16)
                            rgba_color = f'rgba({r},{g},{b},0.2)'
                        else:
                            rgba_color = f'rgba(100,100,100,0.2)'
                        
                        # Add confidence interval band
                        fig_fit.add_trace(go.Scatter(
                            x=np.concatenate([time_data, time_data[::-1]]),
                            y=np.concatenate([fitted_curve_lower, fitted_curve_upper[::-1]]),
                            fill='toself',
                            fillcolor=rgba_color,
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f"{species} 95% CI",
                            hoverinfo='skip'
                        ))
                    fig_fit.add_vrect(
                        x0=time_min, x1=time_max,
                        fillcolor="gray", opacity=0.07,
                        layer="below", line_width=0,
                    )
            fig_fit.update_layout(
                title="Individual Species Data with ODE Fit and 95% Confidence Intervals",
                xaxis_title="Time",
                yaxis_title="Optical Density (OD)",
                template="plotly_white",
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            if log_scale:
                fig_fit.update_yaxes(type="log")
            st.plotly_chart(fig_fit, use_container_width=True)
            st.markdown(get_svg_download_link(fig_fit, "fitted_curves_with_CI"), unsafe_allow_html=True)

        with fit_tabs[2]:
            growth_rates = [r["mu_val"] for r in fit_results]
            species_names = [r["Species"] for r in fit_results]
            fig_rates = go.Figure()
            fig_rates.add_trace(go.Bar(
                x=species_names,
                y=growth_rates,
                text=[f"{mu:.4f}" for mu in growth_rates],
                textposition='auto',
                marker_color=colors[:len(growth_rates)]
            ))
            fig_rates.update_layout(
                title="Comparison of Growth Rates (μ)",
                xaxis_title="Species",
                yaxis_title="Growth Rate (μ)",
                template="plotly_white",
                height=400,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_rates, use_container_width=True)            
            st.markdown(get_svg_download_link(fig_rates, "growth_rate_comparison"), unsafe_allow_html=True)

        with fit_tabs[3]:
            st.subheader("Lotka-Volterra Pairwise Model")
            st.markdown("""
            ### Lotka-Volterra Model for Species Interactions
            The Lotka-Volterra model for pairwise interactions describes how two species grow and interact with each other. 
            The model captures competitive, cooperative, or predator-prey dynamics between species.
            **General Pairwise Lotka-Volterra Equations:**
            """)
            st.latex(r'''
            \begin{align}
            \frac{dx_i}{dt} &= x_i \left( r_i + \sum_{j=1}^n \alpha_{ij} x_j \right) \\
            \end{align}
            ''')
            st.markdown("""
            Where:
            - $x_i$ is the population density of species $i$
            - $r_i$ is the intrinsic growth rate of species $i$
            - $\\alpha_{ij}$ is the interaction coefficient describing the effect of species $j$ on species $i$
                - $\\alpha_{ii}$ is the self-interaction (density-dependent growth) coefficient
                - $\\alpha_{ij}$ (when $i \\neq j$) represents competition ($<0$), mutualism ($>0$), or predation
            **For a two-species system:**
            """)
            st.latex(r'''
            \begin{align}
            \frac{dx_1}{dt} &= x_1(r_1 + \alpha_{11}x_1 + \alpha_{12}x_2) \\
            \frac{dx_2}{dt} &= x_2(r_2 + \alpha_{21}x_1 + \alpha_{22}x_2)
            \end{align}
            ''')
            st.markdown("""
            The signs and magnitudes of the interaction coefficients $\\alpha_{12}$ and $\\alpha_{21}$ determine the type of interaction:
            | Interaction Type | $\\alpha_{12}$ | $\\alpha_{21}$ |
            |------------------|---------------|---------------|
            | Competition      | Negative      | Negative      |
            | Mutualism        | Positive      | Positive      |
            | Predation        | Positive      | Negative      |
            | Commensalism     | Positive      | ≈0            |
            | Amensalism       | Negative      | ≈0            |
            Fitting this model to pairwise co-culture data allows quantification of these interaction strengths.
            """)
            st.info("To fit the Lotka-Volterra model to your data, you'll need both monoculture and co-culture time series. The fitting process involves solving differential equations to estimate the interaction parameters.")

        with fit_tabs[4]:
            st.subheader("Lotka-Volterra Full Species Fit (Pairwise Sum Comparison)")
            
            # Add log scale checkbox for LV fit plots
            lv_log_scale = st.checkbox("Use logarithmic Y-axis for LV plots", value=False, key="lv_log_scale")

            # Reference to the paper
            with st.expander("📚 Reference and Background", expanded=False):
                st.markdown("""
                **Reference:** 
                Barraquand, F., Louca, S., Abbott, K. C., Cobbold, C. A., Cordoleani, F., DeAngelis, D. L., ... & Thomas, L. (2017). 
                Moving forward in circles: challenges and opportunities in modelling population cycles. 
                *Ecology and Evolution*, 7(16), 6106-6131. 
                [DOI: 10.1002/ece3.6926](https://onlinelibrary.wiley.com/doi/epdf/10.1002/ece3.6926)
                
                **Note:** Fitting Lotka-Volterra models is notoriously difficult due to the non-linear nature of the equations 
                and the interdependence of parameters. The optimization can be improved by constraining the signs of 
                interaction coefficients based on biological knowledge.
                """)

            # Load processed data
            df_avg = st.session_state.df_avg
            subtract_bg = st.session_state.subtract_bg
            if subtract_bg:
                data_proc = df_avg.iloc[:, 1:-1].subtract(df_avg["background_avg"], axis=0)
            else:
                data_proc = df_avg.iloc[:, 1:-1]

            species_cols = st.session_state.species_cols
            df_fit = st.session_state.get("df_fit")
            if df_fit is None:
                st.warning("Please run individual species fitting first!")
                st.stop()

            mu_vals = []
            x0_vals = []
            for sp in species_cols:
                mu = df_fit.loc[df_fit["Species"] == sp, "mu_val"].values
                if not len(mu) or np.isnan(mu[0]):
                    st.warning(f"Missing μ for {sp}. Run monoculture fit first.")
                    st.stop()
                mu_vals.append(mu[0])
                x0_vals.append(data_proc[sp].iloc[0])
            mu_vals = np.array(mu_vals)
            x0_vals = np.array(x0_vals)

            time_span = df_avg["time"].values
            n_species = len(species_cols)
            colors = px.colors.qualitative.Bold

            from itertools import combinations
            pair_indices = list(combinations(range(n_species), 2))

            def lv_ode(t, x, mu, A):
                dxdt = np.zeros_like(x)
                for i in range(n_species):
                    dxdt[i] = x[i] * (mu[i] + A[i] @ x)
                return dxdt

            def mse_on_pairs(alpha_flat):
                A = alpha_flat.reshape((n_species, n_species))
                sol = solve_ivp(
                    lv_ode,
                    (time_span[0], time_span[-1]),
                    x0_vals,
                    args=(mu_vals, A),
                    t_eval=time_span,
                    method="RK45"
                )
                if not sol.success:
                    return 1e10
                sim = sol.y
                total = 0.0
                for i, j in pair_indices:
                    obs = data_proc[f"{species_cols[i]}+{species_cols[j]}"].values
                    sim_sum = sim[i] + sim[j]
                    total += np.mean((sim_sum - obs) ** 2)
                return total / len(pair_indices)

            # --- Optional Boundary Configuration ---
            with st.expander("⚙️ Optional: Configure Interaction Sign Constraints", expanded=False):
                st.markdown("""
                **Improve fitting stability by constraining the signs of interaction coefficients:**
                - ✅ **Negative**: Coefficient will be constrained to (-∞, 0)
                - ✅ **Positive**: Coefficient will be constrained to (0, +∞)  
                - ⬜ **Unchecked**: Coefficient can be any value (-10, +10)
                
                **Biological interpretation:**
                - **Diagonal elements** (α_ii): Usually negative (density-dependent growth limitation)
                - **Off-diagonal elements** (α_ij): Negative for competition, positive for mutualism
                """)
                
                # Initialize constraint arrays if not exists
                if "constraint_matrix_negative" not in st.session_state:
                    # Default: diagonal elements negative, off-diagonal unconstrained
                    st.session_state.constraint_matrix_negative = [[i == j for j in range(n_species)] for i in range(n_species)]
                
                if "constraint_matrix_positive" not in st.session_state:
                    st.session_state.constraint_matrix_positive = [[False for j in range(n_species)] for i in range(n_species)]
                
                # Ensure matrices have correct dimensions (in case species count changed)
                if (len(st.session_state.constraint_matrix_negative) != n_species or 
                    len(st.session_state.constraint_matrix_negative[0]) != n_species):
                    st.session_state.constraint_matrix_negative = [[i == j for j in range(n_species)] for i in range(n_species)]
                
                if (len(st.session_state.constraint_matrix_positive) != n_species or 
                    len(st.session_state.constraint_matrix_positive[0]) != n_species):
                    st.session_state.constraint_matrix_positive = [[False for j in range(n_species)] for i in range(n_species)]
                
                # Use a form to prevent constant reruns
                with st.form("constraint_form"):
                    st.markdown("**Configure sign constraints for interaction coefficients:**")
                    
                    # Create constraint matrix UI
                    constraint_neg = []
                    constraint_pos = []
                    
                    for i in range(n_species):
                        st.markdown(f"**Row {i+1}: Effects on {species_cols[i]}**")
                        constraint_cols = st.columns(n_species)
                        row_neg = []
                        row_pos = []
                        
                        for j in range(n_species):
                            with constraint_cols[j]:
                                st.write(f"α({species_cols[i]},{species_cols[j]})")
                                
                                # Negative constraint
                                is_negative = st.checkbox(
                                    "< 0",
                                    value=st.session_state.constraint_matrix_negative[i][j],
                                    key=f"neg_{i}_{j}"
                                )
                                row_neg.append(is_negative)
                                
                                # Positive constraint
                                is_positive = st.checkbox(
                                    "> 0",
                                    value=st.session_state.constraint_matrix_positive[i][j],
                                    key=f"pos_{i}_{j}"
                                )
                                row_pos.append(is_positive)
                        
                        constraint_neg.append(row_neg)
                        constraint_pos.append(row_pos)
                    
                    # Form submit button
                    if st.form_submit_button("Apply Constraints"):
                        st.session_state.constraint_matrix_negative = constraint_neg
                        st.session_state.constraint_matrix_positive = constraint_pos
                        st.success("Constraints updated! Now you can run the LV fit.")
                
                # Show current constraints summary
                st.markdown("**Current applied constraints:**")
                constraint_summary = []
                for i in range(n_species):
                    for j in range(n_species):
                        is_negative = st.session_state.constraint_matrix_negative[i][j]
                        is_positive = st.session_state.constraint_matrix_positive[i][j]
                        
                        if is_negative and not is_positive:
                            constraint_summary.append(f"α({species_cols[i]},{species_cols[j]}) < 0")
                        elif is_positive and not is_negative:
                            constraint_summary.append(f"α({species_cols[i]},{species_cols[j]}) > 0")
                
                if constraint_summary:
                    st.info("Applied constraints: " + ", ".join(constraint_summary))
                else:
                    st.info("No sign constraints applied (all coefficients unconstrained)")

            # --- Run LV Fit Button & Session State Handling ---
            if "lv_fit_done" not in st.session_state:
                st.session_state.lv_fit_done = False
            if "lv_species_sol" not in st.session_state:
                st.session_state.lv_species_sol = None
            if "lv_fit_time" not in st.session_state:
                st.session_state.lv_fit_time = None
            if "A_fit" not in st.session_state:
                st.session_state.A_fit = None
            if "lv_pairwise_metrics" not in st.session_state:
                st.session_state.lv_pairwise_metrics = None
            if "param_errors" not in st.session_state:
                st.session_state.param_errors = None
            if "param_cov_matrix" not in st.session_state:
                st.session_state.param_cov_matrix = None

            init_alpha = np.zeros(n_species * n_species)
            
            # Build bounds based on sign constraints
            bounds = []
            for i in range(n_species):
                for j in range(n_species):
                    is_negative = st.session_state.constraint_matrix_negative[i][j]
                    is_positive = st.session_state.constraint_matrix_positive[i][j]
                    
                    if is_negative and is_positive:
                        st.warning(f"α({species_cols[i]},{species_cols[j]}) cannot be both positive and negative! Using unconstrained.")
                        bounds.append((-10.0, 10.0))
                    elif is_negative:
                        bounds.append((-10.0, -1e-6))  # Negative constraint
                    elif is_positive:
                        bounds.append((1e-6, 10.0))    # Positive constraint
                    else:
                        bounds.append((-10.0, 10.0))   # Unconstrained

            col1, col2 = st.columns([3, 1])
            with col1:
                run_lv = st.button("Run LV Full‐Species Pairwise Sum Fit")
            with col2:
                if st.button("Reset Fit"):
                    st.session_state.lv_fit_done = False
                    st.session_state.lv_species_sol = None
                    st.session_state.lv_fit_time = None
                    st.session_state.A_fit = None
                    st.session_state.lv_pairwise_metrics = None
                    st.session_state.param_errors = None
                    st.session_state.param_cov_matrix = None
                    if "optimization_result" in st.session_state:
                        del st.session_state.optimization_result
                    st.success("Fit results cleared. You can now run with new constraints.")
                    
            if run_lv or st.session_state.lv_fit_done:
                with st.spinner("Fitting α matrix to pairwise sums..."):
                    if not st.session_state.lv_fit_done:
                        res = minimize(
                            mse_on_pairs,
                            init_alpha,
                            method="L-BFGS-B",
                            bounds=bounds
                        )
                        optimization_method = "L-BFGS-B"
                        if not res.success:
                            st.warning("L-BFGS-B failed, trying Nelder-Mead…")
                            # For Nelder-Mead, we need to handle bounds differently
                            # Convert bounds to constraints or use a penalty method
                            res = minimize(
                                mse_on_pairs,
                                init_alpha,
                                method="Nelder-Mead",
                                options={"maxiter": 2000}
                            )
                            optimization_method = "Nelder-Mead (without bounds)"
                        if not res.success:
                            st.error("Both optimizers failed. Try different bounds/initial values.")
                            st.stop()

                        # Save fit results to session state!
                        A_fit = res.x.reshape((n_species, n_species))
                        st.session_state.A_fit = A_fit
                        st.session_state.lv_fit_done = True
                        st.session_state.optimization_result = {
                            "method": optimization_method,
                            "success": res.success,
                            "message": res.message,
                            "nit": res.nit,
                            "fun": res.fun
                        }
                        
                        # Calculate parameter errors from Hessian approximation
                        try:
                            # Approximate Hessian using finite differences
                            from scipy.optimize import approx_fprime
                            eps = np.sqrt(np.finfo(float).eps)
                            
                            def objective_wrapper(params):
                                """Wrapper for objective function"""
                                try:
                                    return mse_on_pairs(params)
                                except:
                                    return 1e10
                            
                            # Calculate Hessian approximation using a more robust method
                            n_params = len(res.x)
                            hessian = np.zeros((n_params, n_params))
                            
                            # Calculate gradient at optimal point
                            grad0 = approx_fprime(res.x, objective_wrapper, eps)
                            
                            # Calculate Hessian by finite differences of gradient
                            for i in range(n_params):
                                x_plus = res.x.copy()
                                x_minus = res.x.copy()
                                x_plus[i] += eps
                                x_minus[i] -= eps
                                
                                grad_plus = approx_fprime(x_plus, objective_wrapper, eps)
                                grad_minus = approx_fprime(x_minus, objective_wrapper, eps)
                                
                                hessian[i] = (grad_plus - grad_minus) / (2 * eps)
                            
                            # Symmetrize the Hessian
                            hessian = 0.5 * (hessian + hessian.T)
                            
                            # Calculate covariance matrix (inverse of Hessian)
                            # Add small diagonal regularization for numerical stability
                            hessian_reg = hessian + 1e-8 * np.eye(n_params)
                            
                            try:
                                # Use pseudo-inverse for better numerical stability
                                cov_matrix = np.linalg.pinv(hessian_reg)
                                # Extract standard errors (diagonal of covariance matrix)
                                param_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
                                
                                # Check if errors are reasonable (not too large)
                                if np.any(param_errors > 1000):
                                    st.session_state.param_errors = np.full((n_species, n_species), np.nan)
                                    st.session_state.param_cov_matrix = None
                                    st.warning("Parameter errors are very large, likely due to poor model fit or numerical instability.")
                                else:
                                    st.session_state.param_errors = param_errors.reshape((n_species, n_species))
                                    st.session_state.param_cov_matrix = cov_matrix
                            except:
                                # If matrix inversion fails, use simpler error estimation
                                st.session_state.param_errors = np.full((n_species, n_species), np.nan)
                                st.session_state.param_cov_matrix = None
                                st.warning("Could not calculate parameter errors due to numerical instability.")
                        except Exception as e:
                            # If error calculation fails, set errors to NaN
                            st.session_state.param_errors = np.full((n_species, n_species), np.nan)
                            st.session_state.param_cov_matrix = None
                            st.warning(f"Error calculation failed: {str(e)}")

                        # Simulate with fitted α
                        sol1 = solve_ivp(
                            lv_ode,
                            (time_span[0], time_span[-1]),
                            x0_vals,
                            args=(mu_vals, A_fit),
                            t_eval=time_span,
                            method="RK45"
                        )
                        sim1 = sol1.y
                        st.session_state.lv_species_sol = sim1
                        st.session_state.lv_fit_time = time_span

                        # Pairwise metrics
                        metrics = []
                        for i, j in pair_indices:
                            name = f"{species_cols[i]}+{species_cols[j]}"
                            obs = data_proc[name].values
                            sim_sum = sim1[i] + sim1[j]
                            err = sim_sum - obs
                            mse = np.mean(err ** 2)
                            rmse = np.sqrt(mse)
                            mae = np.mean(np.abs(err))
                            ss_res = np.sum(err ** 2)
                            ss_tot = np.sum((obs - obs.mean()) ** 2)
                            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                            r = np.corrcoef(obs, sim_sum)[0, 1] if obs.std() > 0 and sim_sum.std() > 0 else np.nan
                            metrics.append({
                                "Pair": name,
                                "MSE": mse,
                                "RMSE": rmse,
                                "MAE": mae,
                                "R²": r2,
                                "Pearson r": r
                            })
                        met_df = pd.DataFrame(metrics)
                        st.session_state.lv_pairwise_metrics = met_df

                # === Plot LV Solution Curves Only ===
                if st.session_state.lv_species_sol is not None:
                    sim_species = st.session_state.lv_species_sol
                    lv_fit_time = st.session_state.lv_fit_time
                    st.markdown("#### Lotka-Volterra Model Solution (Simulated Populations)")
                    fig_lv_sol = go.Figure()
                    
                    # First add confidence intervals if parameter errors are available
                    if (hasattr(st.session_state, 'param_errors') and 
                        st.session_state.param_errors is not None and 
                        st.session_state.param_cov_matrix is not None):
                        
                        # Calculate confidence bands using parameter uncertainties
                        try:
                            # Create upper and lower bound simulations
                            n_samples = 50  # Number of parameter samples for CI calculation
                            
                            # Sample from parameter distribution
                            mean_params = st.session_state.A_fit.flatten()
                            cov_matrix = st.session_state.param_cov_matrix
                            
                            # Generate parameter samples and run simulations
                            all_sim_results = [[] for _ in range(n_species)]
                            
                            for _ in range(n_samples):
                                try:
                                    sample = np.random.multivariate_normal(mean_params, cov_matrix)
                                    A_sample = sample.reshape((n_species, n_species))
                                    
                                    sol_sample = solve_ivp(
                                        lv_ode,
                                        (time_span[0], time_span[-1]),
                                        x0_vals,
                                        args=(mu_vals, A_sample),
                                        t_eval=time_span,
                                        method="RK45"
                                    )
                                    if sol_sample.success:
                                        for sp_idx in range(n_species):
                                            all_sim_results[sp_idx].append(sol_sample.y[sp_idx])
                                except:
                                    continue
                            
                            # Add CI bands for each species (behind the main curves)
                            for idx, sp in enumerate(species_cols):
                                if len(all_sim_results[idx]) > 10:  # Only plot CI if we have enough samples
                                    sim_array = np.array(all_sim_results[idx])
                                    ci_lower = np.percentile(sim_array, 2.5, axis=0)
                                    ci_upper = np.percentile(sim_array, 97.5, axis=0)
                                    
                                    # Convert color to rgba for CI bands
                                    color_hex = colors[idx % len(colors)]
                                    if color_hex.startswith('#'):
                                        r = int(color_hex[1:3], 16)
                                        g = int(color_hex[3:5], 16)
                                        b = int(color_hex[5:7], 16)
                                        rgba_color = f'rgba({r},{g},{b},0.2)'
                                    else:
                                        rgba_color = f'rgba(100,100,100,0.2)'
                                    
                                    # Add confidence interval band
                                    fig_lv_sol.add_trace(go.Scatter(
                                        x=np.concatenate([time_data, time_data[::-1]]),
                                        y=np.concatenate([ci_lower, ci_upper[::-1]]),
                                        fill='toself',
                                        fillcolor=rgba_color,
                                        line=dict(color='rgba(255,255,255,0)'),
                                        showlegend=False,
                                        name=f"{sp} 95% CI",
                                        hoverinfo='skip'
                                    ))
                        except Exception as e:
                            # If CI calculation fails, continue without CI
                            pass
                    
                    # Now add the main fitted curves on top
                    for idx, sp in enumerate(species_cols):
                        fig_lv_sol.add_trace(go.Scatter(
                            x=lv_fit_time, y=sim_species[idx],
                            mode='lines',
                            name=f"{sp} LV fit",
                            line=dict(color=colors[idx % len(colors)], width=3)
                        ))
                    
                    fig_lv_sol.update_layout(
                        title="Lotka-Volterra Model Solution with 95% Confidence Intervals",
                        xaxis_title="Time",
                        yaxis_title="Optical Density (OD)",
                        template="plotly_white",
                        height=500
                    )
                    if lv_log_scale:
                        fig_lv_sol.update_yaxes(type="log")
                    st.plotly_chart(fig_lv_sol, use_container_width=True)

                    # === Show LV Solution Table with SD Error ===
                    st.markdown("#### LV Simulated Populations Table & SD Error")
                    # Calculate error vs data for each species
                    lv_sol_df = pd.DataFrame(sim_species.T, columns=species_cols)
                    lv_sol_df["time"] = lv_fit_time
                    lv_sol_df = lv_sol_df[["time"] + species_cols]
                    err_dict = {}
                    for i, sp in enumerate(species_cols):
                        obs = data_proc[sp].values
                        sim = sim_species[i]
                        sd_err = np.std(sim - obs)
                        err_dict[sp] = sd_err
                    err_table = pd.DataFrame({
                        "Species": list(err_dict.keys()),
                        "SD Error (Sim - Data)": list(err_dict.values())
                    })
                    st.dataframe(lv_sol_df.head(10), use_container_width=True)
                    st.dataframe(err_table, use_container_width=True)

                # === Pairwise Sum Comparison Plot ===
                st.markdown("#### Pairwise Sum: Observed vs Simulated")
                if st.session_state.lv_species_sol is not None:
                    sim_species = st.session_state.lv_species_sol
                    lv_fit_time = st.session_state.lv_fit_time
                    fig2 = go.Figure()
                    
                    # First add confidence intervals for simulated curves if available
                    if (hasattr(st.session_state, 'param_errors') and 
                        st.session_state.param_errors is not None and 
                        st.session_state.param_cov_matrix is not None):
                        
                        try:
                            # Calculate CI for all pairwise sums at once
                            n_samples = 50
                            mean_params = st.session_state.A_fit.flatten()
                            cov_matrix = st.session_state.param_cov_matrix
                            
                            # Store results for each pairwise combination
                            all_pairwise_results = {idx: [] for idx in range(len(pair_indices))}
                            
                            # Generate parameter samples for pairwise sums
                            for _ in range(n_samples):
                                try:
                                    sample = np.random.multivariate_normal(mean_params, cov_matrix)
                                    A_sample = sample.reshape((n_species, n_species))
                                    
                                    sol_sample = solve_ivp(
                                        lv_ode,
                                        (time_span[0], time_span[-1]),
                                        x0_vals,
                                        args=(mu_vals, A_sample),
                                        t_eval=time_span,
                                        method="RK45"
                                    )
                                    if sol_sample.success:
                                        for pair_idx, (i, j) in enumerate(pair_indices):
                                            pairwise_sum = sol_sample.y[i] + sol_sample.y[j]
                                            all_pairwise_results[pair_idx].append(pairwise_sum)
                                except:
                                    continue
                            
                            # Add CI bands for each pairwise sum (behind the main curves)
                            for idx, (i, j) in enumerate(pair_indices):
                                if len(all_pairwise_results[idx]) > 10:  # Only plot CI if we have enough samples
                                    pairwise_array = np.array(all_pairwise_results[idx])
                                    ci_lower = np.percentile(pairwise_array, 2.5, axis=0)
                                    ci_upper = np.percentile(pairwise_array, 97.5, axis=0)
                                    
                                    # Convert color to rgba for CI bands
                                    color = colors[idx % len(colors)]
                                    if color.startswith('#'):
                                        r = int(color[1:3], 16)
                                        g = int(color[3:5], 16)
                                        b = int(color[5:7], 16)
                                        rgba_color = f'rgba({r},{g},{b},0.2)'
                                    else:
                                        rgba_color = f'rgba(100,100,100,0.2)'
                                    
                                    # Add confidence interval band
                                    fig2.add_trace(go.Scatter(
                                        x=np.concatenate([lv_fit_time, lv_fit_time[::-1]]),
                                        y=np.concatenate([ci_lower, ci_upper[::-1]]),
                                        fill='toself',
                                        fillcolor=rgba_color,
                                        line=dict(color='rgba(255,255,255,0)'),
                                        showlegend=False,
                                        name=f"{species_cols[i]}+{species_cols[j]} 95% CI",
                                        hoverinfo='skip'
                                    ))
                        except Exception as e:
                            # If CI calculation fails, continue without CI
                            pass
                    
                    # Now add the main curves on top
                    for idx, (i, j) in enumerate(pair_indices):
                        name = f"{species_cols[i]}+{species_cols[j]}"
                        obs_sum = data_proc[name].values
                        sim_sum = sim_species[i] + sim_species[j]
                        color = colors[idx % len(colors)]
                        
                        # Observed data (no CI)
                        fig2.add_trace(go.Scatter(
                            x=lv_fit_time, y=obs_sum,
                            mode="markers+lines",
                            name=f"Obs {name}",
                            line=dict(color=color),
                            marker=dict(size=4)
                        ))
                        
                        # Simulated data (fitted curve on top of CI)
                        fig2.add_trace(go.Scatter(
                            x=lv_fit_time, y=sim_sum,
                            mode="lines",
                            name=f"Sim {name}",
                            line=dict(color=color, dash="dash", width=3)
                        ))
                    
                    fig2.update_layout(
                        title="Observed vs Simulated Pairwise Sums with 95% Confidence Intervals",
                        xaxis_title="Time", yaxis_title="Population (OD)",
                        template="plotly_white"
                    )
                    if lv_log_scale:
                        fig2.update_yaxes(type="log")
                    st.plotly_chart(fig2, use_container_width=True)

                # === Show Fitted α Matrix ===
                st.markdown("### Fitted interaction matrix (α)")
                
                # Show optimization summary
                if "optimization_result" in st.session_state:
                    opt_result = st.session_state.optimization_result
                    with st.expander("🔍 Optimization Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Method", opt_result["method"])
                        with col2:
                            st.metric("Iterations", opt_result["nit"])
                        with col3:
                            st.metric("Final MSE", f"{opt_result['fun']:.8f}")
                        st.info(f"Optimization status: {opt_result['message']}")
                
                # === Parameter Table with Values and Errors ===
                st.markdown("#### Parameter Table with Standard Errors")
                
                # Create parameter table similar to Model Fitting tab
                if hasattr(st.session_state, 'param_errors') and st.session_state.param_errors is not None:
                    param_table_data = []
                    for i in range(n_species):
                        for j in range(n_species):
                            param_name = f"α({species_cols[i]},{species_cols[j]})"
                            param_value = st.session_state.A_fit[i, j]
                            param_error = st.session_state.param_errors[i, j]
                            
                            # Calculate 95% confidence intervals and significance
                            if not np.isnan(param_error) and param_error > 0:
                                ci_low = param_value - 1.96 * param_error
                                ci_high = param_value + 1.96 * param_error
                                value_str = f"{param_value:.8f} ± {param_error:.8f}"
                                
                                # Check statistical significance (t-test approximation)
                                t_stat = abs(param_value / param_error)
                                # Rough significance indicators
                                if t_stat > 2.58:  # 99% confidence
                                    significance = "***"
                                elif t_stat > 1.96:  # 95% confidence
                                    significance = "**"
                                elif t_stat > 1.645:  # 90% confidence
                                    significance = "*"
                                else:
                                    significance = ""
                            else:
                                ci_low = np.nan
                                ci_high = np.nan
                                value_str = f"{param_value:.8f} ± N/A"
                                significance = ""
                            
                            # Get constraint info
                            is_negative = st.session_state.constraint_matrix_negative[i][j]
                            is_positive = st.session_state.constraint_matrix_positive[i][j]
                            
                            if is_negative and not is_positive:
                                constraint = "< 0"
                            elif is_positive and not is_negative:
                                constraint = "> 0"
                            else:
                                constraint = "free"
                            
                            param_table_data.append({
                                "Parameter": param_name,
                                "Value ± Error": value_str,
                                "Significance": significance,
                                "CI_Low": ci_low,
                                "CI_High": ci_high,
                                "Constraint": constraint,
                                "param_value": param_value,
                                "param_error": param_error
                            })
                    
                    param_df = pd.DataFrame(param_table_data)
                    
                    # Add download button for parameter table
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("Lotka-Volterra Parameters with 95% Confidence Intervals")
                        st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.1")
                    with col2:
                        st.markdown(download_csv(
                            param_df[["Parameter", "Value ± Error", "Significance", "CI_Low", "CI_High", "Constraint"]], 
                            "lv_parameters_with_CI"), unsafe_allow_html=True)
                    
                    # Display parameter table
                    st.dataframe(param_df[[
                        "Parameter", 
                        "Value ± Error", 
                        "Significance",
                        "CI_Low", "CI_High",
                        "Constraint"
                    ]].rename(columns={
                        "CI_Low": "95% CI Low",
                        "CI_High": "95% CI High"
                    }), use_container_width=True)
                else:
                    st.info("Parameter errors not available. This may occur if the Hessian calculation failed.")
                
                # Create a styled dataframe showing the fitted values with constraint info
                alpha_df = pd.DataFrame(st.session_state.A_fit, index=species_cols, columns=species_cols)
                
                # Add constraint information as a separate table
                constraint_info = []
                for i in range(n_species):
                    row_info = []
                    for j in range(n_species):
                        is_negative = st.session_state.constraint_matrix_negative[i][j]
                        is_positive = st.session_state.constraint_matrix_positive[i][j]
                        
                        if is_negative and not is_positive:
                            row_info.append("< 0")
                        elif is_positive and not is_negative:
                            row_info.append("> 0")
                        else:
                            row_info.append("free")
                    constraint_info.append(row_info)
                
                constraint_df = pd.DataFrame(constraint_info, index=species_cols, columns=species_cols)
                
                st.markdown("#### Matrix View")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Fitted α values:**")
                    column_config = {col: st.column_config.NumberColumn(format="%.8f") for col in alpha_df.columns}
                    st.dataframe(alpha_df, use_container_width=True, column_config=column_config)
                with col2:
                    st.write("**Applied constraints:**")
                    st.dataframe(constraint_df, use_container_width=True)

                # === Show Pairwise Sum Fit Metrics ===
                st.markdown("### Pairwise‐Sum Fit Metrics")
                st.dataframe(st.session_state.lv_pairwise_metrics, use_container_width=True)
                st.markdown("**Average across all pairs:**")
                avg = st.session_state.lv_pairwise_metrics.mean(numeric_only=True).to_frame("Average").T
                st.dataframe(avg, use_container_width=True)
                st.success("Lotka-Volterra full species fit completed successfully!")

            else:
                st.info("Press 'Run LV Full‐Species Pairwise Sum Fit' to perform fitting and see results.")




        with fit_tabs[5]:
            st.subheader("Manual Exploration: Species Fits and Pairwise LV Simulation")
            
            # Add log scale checkbox for LV plots
            lv_plots_log_scale = st.checkbox("Use logarithmic Y-axis for LV plots", value=False, key="lv_plots_log_scale")

            n_species = len(species_cols)
            time_span = df_avg["time"].values

            # Defaults for μ
            mu_default = []
            for sp in species_cols:
                mu = df_fit.loc[df_fit["Species"] == sp, "mu_val"].values
                mu_default.append(float(mu[0]) if len(mu) else 0.01)
            mu_default = np.array(mu_default)

            # Defaults for α (from last LV fit)
            if "A_fit" in st.session_state and st.session_state.A_fit is not None:
                alpha_default = st.session_state.A_fit
            else:
                st.warning("No previous LV fit found. α defaults to zeros.")
                alpha_default = np.zeros((n_species, n_species))

            # Initialize manual parameter storage in session state
            if "manual_mu_inputs" not in st.session_state:
                st.session_state.manual_mu_inputs = mu_default.copy()
            if "manual_alpha_inputs" not in st.session_state:
                st.session_state.manual_alpha_inputs = alpha_default.copy()
            
            # Ensure dimensions match current species count - reinitialize if they don't
            if len(st.session_state.manual_mu_inputs) != n_species:
                st.session_state.manual_mu_inputs = mu_default.copy()
            if st.session_state.manual_alpha_inputs.shape != (n_species, n_species):
                st.session_state.manual_alpha_inputs = alpha_default.copy()
            
            # Update α values if new fit results are available
            if "A_fit" in st.session_state and st.session_state.A_fit is not None:
                # Only update if the shapes match and values have changed
                if (st.session_state.manual_alpha_inputs.shape == st.session_state.A_fit.shape and 
                    not np.allclose(st.session_state.manual_alpha_inputs, st.session_state.A_fit, rtol=1e-10)):
                    st.session_state.manual_alpha_inputs = st.session_state.A_fit.copy()
                    st.info("✅ Interaction matrix updated with values from LV fit!")

            # Add manual refresh button for α matrix
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("#### Parameter Configuration")
            with col2:
                if st.button("🔄 Sync α from LV Fit", help="Update α matrix with latest values from LV fit"):
                    if "A_fit" in st.session_state and st.session_state.A_fit is not None:
                        st.session_state.manual_alpha_inputs = st.session_state.A_fit.copy()
                        st.success("✅ α matrix synchronized with LV fit results!")
                    else:
                        st.warning("No LV fit results found. Run LV fit first.")

            # Parameter input form to prevent constant reruns
            with st.form("manual_parameters_form"):
                st.markdown("#### Configure Parameters for Manual Exploration")
                
                # μ controls
                st.markdown("**Growth Rates (μ)**")
                mu_inputs = []
                mu_cols = st.columns(n_species)
                for i, sp in enumerate(species_cols):
                    with mu_cols[i]:
                        # Ensure index is within bounds
                        default_mu = (float(st.session_state.manual_mu_inputs[i]) 
                                    if i < len(st.session_state.manual_mu_inputs) 
                                    else mu_default[i])
                        mu_val = st.number_input(
                            f"μ for {sp}",
                            value=default_mu,
                            format="%.8f",
                            key=f"form_manual_mu_{sp}"
                        )
                        mu_inputs.append(mu_val)
                mu_inputs = np.array(mu_inputs)

                # α controls
                st.markdown("**Interaction Matrix (α)**")
                alpha_inputs = []
                for i in range(n_species):
                    st.markdown(f"*Row {i+1}: Effects on {species_cols[i]}*")
                    alpha_cols = st.columns(n_species)
                    row = []
                    for j in range(n_species):
                        with alpha_cols[j]:
                            # Ensure indices are within bounds
                            default_alpha = (float(st.session_state.manual_alpha_inputs[i, j])
                                           if (i < st.session_state.manual_alpha_inputs.shape[0] and 
                                               j < st.session_state.manual_alpha_inputs.shape[1])
                                           else alpha_default[i, j])
                            val = st.number_input(
                                f"α({species_cols[i]},{species_cols[j]})",
                                value=default_alpha,
                                format="%.8f",
                                key=f"form_manual_alpha_{i}_{j}"
                            )
                            row.append(val)
                    alpha_inputs.append(row)
                alpha_inputs = np.array(alpha_inputs)

                # Form submit button
                if st.form_submit_button("Update Parameters and Plots"):
                    st.session_state.manual_mu_inputs = mu_inputs.copy()
                    st.session_state.manual_alpha_inputs = alpha_inputs.copy()
                    st.success("Parameters updated! Plots will refresh with new values.")

            # Use the stored parameters for plotting
            mu_inputs = st.session_state.manual_mu_inputs
            alpha_inputs = st.session_state.manual_alpha_inputs

            # --- INDIVIDUAL SPECIES FIT (Exponential, using μ) ---
            st.markdown("### Individual Species Exponential Fit (no interactions)")

            fig_exp = go.Figure()
            for i, sp in enumerate(species_cols):
                # Observed data
                fig_exp.add_trace(go.Scatter(
                    x=time_span, y=data_proc[sp].values,
                    mode="markers", name=f"{sp} obs"
                ))
                # Exponential fit curve (using chosen μ)
                x0 = data_proc[sp].iloc[0]
                exp_pred = x0 * np.exp(mu_inputs[i] * time_span)
                fig_exp.add_trace(go.Scatter(
                    x=time_span, y=exp_pred,
                    mode="lines", name=f"{sp} exp fit (μ={mu_inputs[i]:.4f})"
                ))
            fig_exp.update_layout(
                title="Individual Exponential Fit (μ only, α=0)",
                xaxis_title="Time",
                yaxis_title="Population (OD)",
                template="plotly_white"
            )
            if lv_plots_log_scale:
                fig_exp.update_yaxes(type="log")
            st.plotly_chart(fig_exp, use_container_width=True)

            # --- PAIRWISE SUMS, LV SIMULATION (using μ and α) ---
            st.markdown("### Lotka-Volterra Pairwise Simulation (using μ and α)")

            from itertools import combinations
            pair_indices = list(combinations(range(n_species), 2))
            x0_vals = np.array([data_proc[sp].iloc[0] for sp in species_cols])

            def manual_lv_ode(t, x):
                dxdt = np.zeros_like(x)
                for i in range(n_species):
                    dxdt[i] = x[i] * (mu_inputs[i] + np.dot(alpha_inputs[i], x))
                return dxdt

            sol = solve_ivp(
                manual_lv_ode,
                (time_span[0], time_span[-1]),
                x0_vals,
                t_eval=time_span,
                method="RK45"
            )

            fig_pairwise = go.Figure()
            colors = px.colors.qualitative.Bold
            for idx, (i, j) in enumerate(pair_indices):
                name = f"{species_cols[i]}+{species_cols[j]}"
                obs_sum = data_proc[name].values
                sim_sum = sol.y[i] + sol.y[j]
                color = colors[idx % len(colors)]
                fig_pairwise.add_trace(go.Scatter(
                    x=time_span, y=obs_sum,
                    mode="markers", name=f"Obs {name}",
                    marker=dict(color=color)
                ))
                fig_pairwise.add_trace(go.Scatter(
                    x=time_span, y=sim_sum,
                    mode="lines", name=f"LV sim {name}",
                    line=dict(color=color, dash="dash")
                ))
            fig_pairwise.update_layout(
                title="Pairwise Sums: Data vs Lotka-Volterra Simulation",
                xaxis_title="Time",
                yaxis_title="Population (OD)",
                template="plotly_white"
            )
            if lv_plots_log_scale:
                fig_pairwise.update_yaxes(type="log")
            st.plotly_chart(fig_pairwise, use_container_width=True)

            # --- Phase Plane Configuration ---
            st.markdown("### Phase Plane with Multiple Initial Conditions")

            # Initialize phase plane settings in session state
            if "phase_plane_settings" not in st.session_state:
                st.session_state.phase_plane_settings = {
                    "species_selection": [0, 1] if n_species >= 2 else [0],
                    "x0_min": 0.5,  # Initial condition ranges
                    "x0_max": 5.0,
                    "y0_min": 0.5,
                    "y0_max": 5.0,
                    "plot_x_min": 0.0,  # Plot viewing ranges
                    "plot_x_max": 10.0,
                    "plot_y_min": 0.0,
                    "plot_y_max": 10.0,
                    "show_nullclines": False,  # Visual options
                    "show_vector_field": True,
                    "n_curves": 12
                }

            # Phase plane configuration form
            with st.form("phase_plane_form"):
                st.markdown("**Configure Phase Plane Settings**")
                
                # Species selection
                phase_plane_options = []
                species_cols_for_phase = st.columns(n_species)
                for i, sp in enumerate(species_cols):
                    with species_cols_for_phase[i]:
                        selected = st.checkbox(
                            f"Phase plane: {sp}",
                            value=(i in st.session_state.phase_plane_settings["species_selection"]),
                            key=f"form_phase_plane_select_{sp}"
                        )
                        if selected:
                            phase_plane_options.append(i)
                
                # Range settings (only show if exactly 2 species selected)
                if len(phase_plane_options) == 2:
                    i, j = phase_plane_options
                    sp_i, sp_j = species_cols[i], species_cols[j]
                    
                    # Create tabs to separate initial conditions from plot viewing area
                    init_tab, plot_tab = st.tabs(["🎯 Initial Conditions", "📊 Plot Viewing Area"])
                    
                    with init_tab:
                        st.markdown("**Starting points for trajectories:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            x0_min = st.number_input(f"Initial {sp_i} min", 
                                                   value=st.session_state.phase_plane_settings.get("x0_min", 0.5), 
                                                   step=0.1, key="phase_x0_min")
                            x0_max = st.number_input(f"Initial {sp_i} max", 
                                                   value=st.session_state.phase_plane_settings.get("x0_max", 5.0), 
                                                   step=0.1, key="phase_x0_max")
                        with col2:
                            y0_min = st.number_input(f"Initial {sp_j} min", 
                                                   value=st.session_state.phase_plane_settings.get("y0_min", 0.5), 
                                                   step=0.1, key="phase_y0_min")
                            y0_max = st.number_input(f"Initial {sp_j} max", 
                                                   value=st.session_state.phase_plane_settings.get("y0_max", 5.0), 
                                                   step=0.1, key="phase_y0_max")
                    
                    with plot_tab:
                        st.markdown("**Axis ranges for phase plot display:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            plot_x_min = st.number_input(f"Plot {sp_i} min", 
                                                       value=st.session_state.phase_plane_settings.get("plot_x_min", 0.0), 
                                                       step=0.1, key="phase_plot_x_min")
                            plot_x_max = st.number_input(f"Plot {sp_i} max", 
                                                       value=st.session_state.phase_plane_settings.get("plot_x_max", 10.0), 
                                                       step=0.1, key="phase_plot_x_max")
                        with col2:
                            plot_y_min = st.number_input(f"Plot {sp_j} min", 
                                                       value=st.session_state.phase_plane_settings.get("plot_y_min", 0.0), 
                                                       step=0.1, key="phase_plot_y_min")
                            plot_y_max = st.number_input(f"Plot {sp_j} max", 
                                                       value=st.session_state.phase_plane_settings.get("plot_y_max", 10.0), 
                                                       step=0.1, key="phase_plot_y_max")
                        
                        # Visual options
                        st.markdown("**Visual Options:**")
                        show_nullclines = st.checkbox("Show Nullclines", 
                                                    value=st.session_state.phase_plane_settings.get("show_nullclines", False),
                                                    help="Show lines where dx/dt=0 and dy/dt=0", 
                                                    key="phase_show_nullclines")
                        show_vector_field = st.checkbox("Show Vector Field", 
                                                       value=st.session_state.phase_plane_settings.get("show_vector_field", True),
                                                       help="Show directional arrows", 
                                                       key="phase_show_vectors")
                    
                    n_curves = st.slider("Number of initial conditions", 5, 50, 
                                        st.session_state.phase_plane_settings.get("n_curves", 12))
                else:
                    # Default values if not exactly 2 species selected
                    x0_min, x0_max = st.session_state.phase_plane_settings.get("x0_min", 0.5), st.session_state.phase_plane_settings.get("x0_max", 5.0)
                    y0_min, y0_max = st.session_state.phase_plane_settings.get("y0_min", 0.5), st.session_state.phase_plane_settings.get("y0_max", 5.0)
                    plot_x_min, plot_x_max = st.session_state.phase_plane_settings.get("plot_x_min", 0.0), st.session_state.phase_plane_settings.get("plot_x_max", 10.0)
                    plot_y_min, plot_y_max = st.session_state.phase_plane_settings.get("plot_y_min", 0.0), st.session_state.phase_plane_settings.get("plot_y_max", 10.0)
                    show_nullclines = st.session_state.phase_plane_settings.get("show_nullclines", False)
                    show_vector_field = st.session_state.phase_plane_settings.get("show_vector_field", True)
                    n_curves = st.session_state.phase_plane_settings.get("n_curves", 12)
                
                # Form submit button (always present)
                if st.form_submit_button("Update Phase Plane"):
                    if len(phase_plane_options) == 2:
                        st.session_state.phase_plane_settings = {
                            "species_selection": phase_plane_options,
                            "x0_min": x0_min, "x0_max": x0_max,
                            "y0_min": y0_min, "y0_max": y0_max,
                            "plot_x_min": plot_x_min, "plot_x_max": plot_x_max,
                            "plot_y_min": plot_y_min, "plot_y_max": plot_y_max,
                            "show_nullclines": show_nullclines,
                            "show_vector_field": show_vector_field,
                            "n_curves": n_curves
                        }
                        st.success("Phase plane settings updated!")
                    else:
                        st.session_state.phase_plane_settings["species_selection"] = phase_plane_options
                        if len(phase_plane_options) != 2:
                            st.warning("Please select exactly two species for phase plane visualization.")

            # Display phase plane plot
            phase_settings = st.session_state.phase_plane_settings
            if len(phase_settings["species_selection"]) != 2:
                st.info("Select exactly two species for phase plane visualization.")
            else:
                i, j = phase_settings["species_selection"]
                sp_i, sp_j = species_cols[i], species_cols[j]

                # Define the 2D LV dynamics for the selected species
                def lv_dynamics_2d(t, x, u, params):
                    full_state = np.zeros(n_species)
                    full_state[i] = x[0]
                    full_state[j] = x[1]
                    
                    # Assume other species are at their initial value (or zero if not applicable)
                    for k in range(n_species):
                        if k != i and k != j:
                            full_state[k] = x0_vals[k]

                    dxdt_full = full_state * (mu_inputs + alpha_inputs @ full_state)
                    return [dxdt_full[i], dxdt_full[j]]

                with st.spinner("Generating phase portrait..."):
                    try:
                        # Create Plotly figure for phase portrait
                        fig_phase = go.Figure()
                        
                        # Generate initial conditions for trajectories
                        x0_trajectories = []
                        if phase_settings['n_curves'] > 0:
                            # Create a grid of starting points
                            num_points_per_axis = int(np.ceil(np.sqrt(phase_settings['n_curves'])))
                            x_points = np.linspace(phase_settings['x0_min'], phase_settings['x0_max'], num_points_per_axis)
                            y_points = np.linspace(phase_settings['y0_min'], phase_settings['y0_max'], num_points_per_axis)
                            for x_val in x_points:
                                for y_val in y_points:
                                    x0_trajectories.append([x_val, y_val])

                        # Create phase portrait with vector field using Plotly
                        # Set up grid for vector field using plot ranges instead of initial value ranges
                        x_range = np.linspace(phase_settings.get('plot_x_min', phase_settings['x0_min']), 
                                             phase_settings.get('plot_x_max', phase_settings['x0_max']), 12)
                        y_range = np.linspace(phase_settings.get('plot_y_min', phase_settings['y0_min']), 
                                             phase_settings.get('plot_y_max', phase_settings['y0_max']), 12)
                        X_grid, Y_grid = np.meshgrid(x_range, y_range)
                        
                        # Calculate vector field
                        DX = np.zeros_like(X_grid)
                        DY = np.zeros_like(Y_grid)
                        
                        for xi in range(len(x_range)):
                            for yi in range(len(y_range)):
                                try:
                                    derivatives = lv_dynamics_2d(0, [X_grid[yi, xi], Y_grid[yi, xi]], None, None)
                                    DX[yi, xi] = derivatives[0]
                                    DY[yi, xi] = derivatives[1]
                                except:
                                    DX[yi, xi] = 0
                                    DY[yi, xi] = 0
                        
                        # Add vector field arrows using Plotly (CONDITIONAL)
                        if phase_settings.get("show_vector_field", True):
                            # Normalize arrows for better visualization
                            magnitude = np.sqrt(DX**2 + DY**2)
                            max_mag = np.max(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1.0
                            
                            if max_mag > 0:
                                # Scale arrows appropriately
                                scale_factor = 0.3 * min(phase_settings.get('plot_x_max', phase_settings['x0_max']) - phase_settings.get('plot_x_min', phase_settings['x0_min']), 
                                                       phase_settings.get('plot_y_max', phase_settings['y0_max']) - phase_settings.get('plot_y_min', phase_settings['y0_min'])) / len(x_range)
                            
                            for xi in range(0, len(x_range), 2):  # Skip some arrows for clarity
                                for yi in range(0, len(y_range), 2):
                                    if magnitude[yi, xi] > 0:
                                        x_pos = X_grid[yi, xi]
                                        y_pos = Y_grid[yi, xi]
                                        
                                        # Normalize and scale
                                        dx_norm = DX[yi, xi] / max_mag * scale_factor
                                        dy_norm = DY[yi, xi] / max_mag * scale_factor
                                        
                                        # Add arrow line
                                        fig_phase.add_trace(go.Scatter(
                                            x=[x_pos, x_pos + dx_norm],
                                            y=[y_pos, y_pos + dy_norm],
                                            mode='lines',
                                            line=dict(color='gray', width=1.5),
                                            showlegend=False,
                                            hoverinfo='text',
                                            hovertext=f"Vector at ({x_pos:.2f}, {y_pos:.2f})<br>dx/dt = {DX[yi, xi]:.3f}<br>dy/dt = {DY[yi, xi]:.3f}"
                                        ))
                                        
                                        # Add arrowhead
                                        if abs(dx_norm) > 1e-10 or abs(dy_norm) > 1e-10:
                                            arrow_length = scale_factor * 0.3
                                            arrow_angle = np.pi / 6  # 30 degrees
                                            
                                            # Calculate arrow direction
                                            arrow_dir = np.arctan2(dy_norm, dx_norm)
                                            
                                            # Arrowhead points
                                            x_end = x_pos + dx_norm
                                            y_end = y_pos + dy_norm
                                            
                                            arrow_x1 = x_end - arrow_length * np.cos(arrow_dir - arrow_angle)
                                            arrow_y1 = y_end - arrow_length * np.sin(arrow_dir - arrow_angle)
                                            arrow_x2 = x_end - arrow_length * np.cos(arrow_dir + arrow_angle)
                                            arrow_y2 = y_end - arrow_length * np.sin(arrow_dir + arrow_angle)
                                            
                                            # Add arrowhead lines
                                            fig_phase.add_trace(go.Scatter(
                                                x=[x_end, arrow_x1],
                                                y=[y_end, arrow_y1],
                                                mode='lines',
                                                line=dict(color='gray', width=1.5),
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ))
                                            fig_phase.add_trace(go.Scatter(
                                                x=[x_end, arrow_x2],
                                                y=[y_end, arrow_y2],
                                                mode='lines',
                                                line=dict(color='gray', width=1.5),
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ))
                        
                        # Add nullclines (where dx/dt = 0 or dy/dt = 0) - CONDITIONAL
                        if phase_settings.get("show_nullclines", False):
                            try:
                                # Use plot ranges for nullclines instead of initial value ranges
                                x_fine = np.linspace(phase_settings.get('plot_x_min', phase_settings['x0_min']), 
                                                   phase_settings.get('plot_x_max', phase_settings['x0_max']), 100)
                                y_fine = np.linspace(phase_settings.get('plot_y_min', phase_settings['y0_min']), 
                                                   phase_settings.get('plot_y_max', phase_settings['y0_max']), 100)
                                X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
                                
                                DX_fine = np.zeros_like(X_fine)
                                DY_fine = np.zeros_like(Y_fine)
                                
                                # Calculate derivatives with better error handling
                                valid_points = 0
                                for xi in range(len(x_fine)):
                                    for yi in range(len(y_fine)):
                                        try:
                                            # Ensure positive values to avoid domain errors
                                            x_val = max(1e-10, X_fine[yi, xi])
                                            y_val = max(1e-10, Y_fine[yi, xi])
                                            derivatives = lv_dynamics_2d(0, [x_val, y_val], None, None)
                                            if not (np.isnan(derivatives[0]) or np.isnan(derivatives[1]) or 
                                                   np.isinf(derivatives[0]) or np.isinf(derivatives[1])):
                                                DX_fine[yi, xi] = derivatives[0]
                                                DY_fine[yi, xi] = derivatives[1]
                                                valid_points += 1
                                            else:
                                                DX_fine[yi, xi] = np.nan
                                                DY_fine[yi, xi] = np.nan
                                        except:
                                            DX_fine[yi, xi] = np.nan
                                            DY_fine[yi, xi] = np.nan
                                
                                # Only add nullclines if we have enough valid points and reasonable dynamics
                                if valid_points > len(x_fine) * len(y_fine) * 0.5:  # At least 50% valid points
                                    # Filter out extreme values that can make ugly plots
                                    dx_std = np.nanstd(DX_fine)
                                    dy_std = np.nanstd(DY_fine)
                                    dx_threshold = 5 * dx_std if dx_std > 0 else 1
                                    dy_threshold = 5 * dy_std if dy_std > 0 else 1
                                    
                                    # Mask extreme values
                                    DX_fine[np.abs(DX_fine) > dx_threshold] = np.nan
                                    DY_fine[np.abs(DY_fine) > dy_threshold] = np.nan
                                    
                                    # Add cleaner nullclines
                                    fig_phase.add_trace(go.Contour(
                                        x=x_fine,
                                        y=y_fine,
                                        z=DX_fine,
                                        contours=dict(
                                            start=0, end=0, size=1,
                                            coloring='lines'  # Only show lines, not filled contours
                                        ),
                                        line=dict(color='darkred', width=2),
                                        showscale=False,
                                        showlegend=False,
                                        hoverinfo='text',
                                        hovertext=f'{sp_i} nullcline (d{sp_i}/dt = 0)',
                                        name=f'{sp_i} nullcline'
                                    ))
                                    
                                    fig_phase.add_trace(go.Contour(
                                        x=x_fine,
                                        y=y_fine,
                                        z=DY_fine,
                                        contours=dict(
                                            start=0, end=0, size=1,
                                            coloring='lines'  # Only show lines, not filled contours
                                        ),
                                        line=dict(color='darkblue', width=2),
                                        showscale=False,
                                        showlegend=False,
                                        hoverinfo='text',
                                        hovertext=f'{sp_j} nullcline (d{sp_j}/dt = 0)',
                                        name=f'{sp_j} nullcline'
                                    ))
                                else:
                                    # Skip nullclines if calculation is unreliable
                                    pass
                                
                            except Exception as nullcline_error:
                                # If nullcline calculation fails, continue without them - no error message
                                pass
                        
                        # Add trajectories if requested
                        if x0_trajectories:
                            trajectory_colors = px.colors.qualitative.Set1
                            for idx, x0_point in enumerate(x0_trajectories[:phase_settings['n_curves']]):
                                try:
                                    # Simulate individual trajectory
                                    sol_traj = solve_ivp(
                                        lambda t, x: lv_dynamics_2d(t, x, None, None),
                                        (time_span[0], time_span[-1]),
                                        x0_point,
                                        t_eval=np.linspace(time_span[0], time_span[-1], 100),
                                        method="RK45"
                                    )
                                    if sol_traj.success and len(sol_traj.y[0]) > 1:
                                        color = trajectory_colors[idx % len(trajectory_colors)]
                                        
                                        # Add trajectory line
                                        fig_phase.add_trace(go.Scatter(
                                            x=sol_traj.y[0],
                                            y=sol_traj.y[1],
                                            mode='lines',
                                            line=dict(color=color, width=2),
                                            showlegend=False,
                                            hoverinfo='text',
                                            hovertext=f"Trajectory from ({x0_point[0]:.2f}, {x0_point[1]:.2f})",
                                            name=f"Trajectory {idx+1}"
                                        ))
                                        
                                        # Add starting point
                                        fig_phase.add_trace(go.Scatter(
                                            x=[x0_point[0]],
                                            y=[x0_point[1]],
                                            mode='markers',
                                            marker=dict(color=color, size=8, symbol='circle'),
                                            showlegend=False,
                                            hoverinfo='text',
                                            hovertext=f"Start: ({x0_point[0]:.2f}, {x0_point[1]:.2f})",
                                            name=f"Start {idx+1}"
                                        ))
                                except Exception as traj_error:
                                    continue
                        
                        # Update layout
                        fig_phase.update_layout(
                            title=f"Phase Portrait: {sp_i} vs {sp_j}<br><sub>Red: {sp_i} nullclines, Blue: {sp_j} nullclines, Gray: Vector field</sub>",
                            xaxis_title=f"{sp_i} Population",
                            yaxis_title=f"{sp_j} Population",
                            template="plotly_white",
                            height=600,
                            showlegend=False,
                            hovermode='closest'
                        )
                        
                        # Set axis ranges with proper aspect ratio
                        x_range = phase_settings.get('plot_x_max', phase_settings['x0_max']) - phase_settings.get('plot_x_min', phase_settings['x0_min'])
                        y_range = phase_settings.get('plot_y_max', phase_settings['y0_max']) - phase_settings.get('plot_y_min', phase_settings['y0_min'])
                        
                        fig_phase.update_xaxes(
                            range=[phase_settings.get('plot_x_min', phase_settings['x0_min']), 
                                   phase_settings.get('plot_x_max', phase_settings['x0_max'])],
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            scaleanchor="y",  # Lock aspect ratio
                            scaleratio=1      # 1:1 aspect ratio
                        )
                        fig_phase.update_yaxes(
                            range=[phase_settings.get('plot_y_min', phase_settings['y0_min']), 
                                   phase_settings.get('plot_y_max', phase_settings['y0_max'])],
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray'
                        )
                        
                        # Calculate appropriate dimensions to prevent stretching
                        aspect_ratio = x_range / y_range if y_range > 0 else 1.0
                        base_width = 700
                        plot_height = int(base_width / aspect_ratio) if aspect_ratio > 0 else 600
                        plot_height = max(400, min(800, plot_height))  # Limit height bounds
                        
                        fig_phase.update_layout(width=base_width, height=plot_height)
                        
                        st.plotly_chart(fig_phase, use_container_width=False)
                        
                        # Download link for the plot
                        st.markdown(get_svg_download_link(fig_phase, f"phase_portrait_{sp_i}_vs_{sp_j}"), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"An error occurred while generating the phase plot: {e}")
                        st.error(f"Error details: {str(e)}")  # More detailed error info

            # --- QUIVER PLOT: VECTOR FIELD VISUALIZATION ---
            st.markdown("### Quiver Plot: Vector Field Visualization")
            
            st.markdown("""
            **Quiver plots** show the direction and magnitude of population changes at different points in phase space. 
            Each arrow represents the derivative (dx/dt, dy/dt) at that point, helping visualize the flow of the dynamical system.
            
            💡 **Setup Guide:**
            - **Initial Value Ranges**: Control where sample trajectories start (like phase plane)
            - **Grid Ranges**: Control the region where vector field arrows are displayed
            - Typically, set grid ranges wider than initial value ranges to see the full flow pattern
            """)

            # Initialize quiver plot settings in session state with backward compatibility
            # Reset quiver_settings if it has old structure
            if "quiver_settings" in st.session_state:
                if "x_range" in st.session_state.quiver_settings or "x0_min" not in st.session_state.quiver_settings:
                    # Old structure detected, reset with new structure
                    old_settings = st.session_state.quiver_settings
                    st.session_state.quiver_settings = {
                        "species_selection": old_settings.get("species_selection", [0, 1] if n_species >= 2 else [0]),
                        "x0_min": 0.5,  # Initial value range for species x
                        "x0_max": 5.0,
                        "y0_min": 0.5,  # Initial value range for species y  
                        "y0_max": 5.0,
                        "grid_x_min": old_settings.get("x_range", [0.1, 10.0])[0] if "x_range" in old_settings else 0.1,
                        "grid_x_max": old_settings.get("x_range", [0.1, 10.0])[1] if "x_range" in old_settings else 10.0,
                        "grid_y_min": old_settings.get("y_range", [0.1, 10.0])[0] if "y_range" in old_settings else 0.1,
                        "grid_y_max": old_settings.get("y_range", [0.1, 10.0])[1] if "y_range" in old_settings else 10.0,
                        "grid_resolution": old_settings.get("grid_resolution", 15),
                        "arrow_scale": old_settings.get("arrow_scale", 1.0),
                        "show_nullclines": old_settings.get("show_nullclines", True),
                        "show_equilibrium": old_settings.get("show_equilibrium", True),
                        "show_trajectories": old_settings.get("show_trajectories", True),
                        "n_curves": old_settings.get("n_curves", 12)
                    }
                    st.info("🔄 Quiver plot settings updated to new format!")
                else:
                    # Ensure all required keys exist (safety check)
                    required_keys = ["species_selection", "x0_min", "x0_max", "y0_min", "y0_max", "n_curves"]
                    for key in required_keys:
                        if key not in st.session_state.quiver_settings:
                            if key == "species_selection":
                                st.session_state.quiver_settings[key] = [0, 1] if n_species >= 2 else [0]
                            elif key in ["x0_min", "y0_min"]:
                                st.session_state.quiver_settings[key] = 0.5
                            elif key in ["x0_max", "y0_max"]:
                                st.session_state.quiver_settings[key] = 5.0
                            elif key == "n_curves":
                                st.session_state.quiver_settings[key] = 12
            else:
                # Initialize fresh
                st.session_state.quiver_settings = {
                    "species_selection": [0, 1] if n_species >= 2 else [0],
                    "x0_min": 0.5,  # Initial value range for species x
                    "x0_max": 5.0,
                    "y0_min": 0.5,  # Initial value range for species y  
                    "y0_max": 5.0,
                    "grid_x_min": 0.1,  # Grid range for vector field
                    "grid_x_max": 10.0,
                    "grid_y_min": 0.1,
                    "grid_y_max": 10.0,
                    "grid_resolution": 15,
                    "arrow_scale": 1.0,
                    "show_nullclines": True,
                    "show_equilibrium": True,
                    "show_trajectories": True,
                    "n_curves": 12
                }

            # Quiver plot configuration form
            if len(st.session_state.quiver_settings['species_selection']) == 2:
                i, j = st.session_state.quiver_settings['species_selection']
                sp_i, sp_j = species_cols[i], species_cols[j]
                st.markdown(f"**Auto-Detect Ranges for {sp_i} & {sp_j}**")
                st.info("This will set the grid ranges below based on the min/max values in your data.")
                if st.button("🔍 Apply Auto-Detected Ranges"):
                    # Find all relevant columns
                    relevant_cols = [sp_i, sp_j]
                    for col in data_proc.columns:
                        if sp_i in col.split('+') or sp_j in col.split('+'):
                            if col not in relevant_cols:
                                relevant_cols.append(col)
                    
                    # Get min/max for each species from the relevant data
                    x_min = data_proc[relevant_cols].min().min()
                    x_max = data_proc[relevant_cols].max().max()
                    
                    padding = (x_max - x_min) * 0.2
                    
                    st.session_state.quiver_settings["grid_x_min"] = float(max(0, x_min - padding))
                    st.session_state.quiver_settings["grid_x_max"] = float(x_max + padding)
                    st.session_state.quiver_settings["grid_y_min"] = float(max(0, x_min - padding))
                    st.session_state.quiver_settings["grid_y_max"] = float(x_max + padding)
                    
                    st.success(f"Auto-detected ranges applied. You can now update the plot.")
                    st.experimental_rerun()

            with st.form("quiver_form"):
                st.markdown("**Configure Quiver Plot Settings**")
                
                # Species selection for quiver plot
                quiver_species_options = []
                quiver_species_cols = st.columns(n_species)
                for i, sp in enumerate(species_cols):
                    with quiver_species_cols[i]:
                        selected = st.checkbox(
                            f"Quiver: {sp}",
                            value=(i in st.session_state.quiver_settings["species_selection"]),
                            key=f"form_quiver_select_{sp}"
                        )
                        if selected:
                            quiver_species_options.append(i)
                
                if len(quiver_species_options) == 2:
                    i, j = quiver_species_options
                    sp_i, sp_j = species_cols[i], species_cols[j]
                    
                    # Tab system for cleaner organization
                    tab1, tab2 = st.tabs(["📍 Grid & Trajectory", "⚙️ Display Options"])
                    
                    with tab1:
                        st.markdown(f"**Grid Range for {sp_i} (X-axis)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.number_input(f"Grid {sp_i} min", value=float(st.session_state.quiver_settings.get("grid_x_min", 0.1)), step=0.1, min_value=0.0, key="form_quiver_grid_x_min")
                        with col2:
                            st.number_input(f"Grid {sp_i} max", value=float(st.session_state.quiver_settings.get("grid_x_max", 10.0)), step=0.1, min_value=0.1, key="form_quiver_grid_x_max")

                        st.markdown(f"**Grid Range for {sp_j} (Y-axis)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.number_input(f"Grid {sp_j} min", value=float(st.session_state.quiver_settings.get("grid_y_min", 0.1)), step=0.1, min_value=0.0, key="form_quiver_grid_y_min")
                        with col2:
                            st.number_input(f"Grid {sp_j} max", value=float(st.session_state.quiver_settings.get("grid_y_max", 10.0)), step=0.1, min_value=0.1, key="form_quiver_grid_y_max")

                        st.markdown("**Sample Trajectory Initial Values**")
                        st.slider("Number of sample trajectories", 0, 20, st.session_state.quiver_settings.get("n_curves", 12), key="form_quiver_n_curves")

                    with tab2:
                        st.markdown("**Visual Options**")
                        st.slider("Vector Grid Resolution", 8, 30, st.session_state.quiver_settings.get("grid_resolution", 15), help="Number of arrows per axis", key="form_quiver_grid_res")
                        st.checkbox("Show Nullclines", value=st.session_state.quiver_settings.get("show_nullclines", True), key="form_quiver_nullclines")
                
                if st.form_submit_button("🔄 Update Quiver Plot"):
                    st.session_state.quiver_settings["species_selection"] = quiver_species_options
                    if len(quiver_species_options) == 2:
                        st.session_state.quiver_settings.update({
                            "grid_x_min": st.session_state.form_quiver_grid_x_min,
                            "grid_x_max": st.session_state.form_quiver_grid_x_max,
                            "grid_y_min": st.session_state.form_quiver_grid_y_min,
                            "grid_y_max": st.session_state.form_quiver_grid_y_max,
                            "n_curves": st.session_state.form_quiver_n_curves,
                            "grid_resolution": st.session_state.form_quiver_grid_res,
                            "show_nullclines": st.session_state.form_quiver_nullclines
                        })
                        st.success("✅ Quiver plot settings updated!")
                    else:
                        st.warning("Please select exactly two species for quiver plot visualization.")


            # Display quiver plot
            quiver_settings = st.session_state.quiver_settings
            if len(quiver_settings["species_selection"]) != 2:
                st.info("Select exactly two species for quiver plot visualization.")
            else:
                i, j = quiver_settings["species_selection"]
                sp_i, sp_j = species_cols[i], species_cols[j]

                # Define the 2D LV dynamics for the selected species
                x0_vals = np.array([data_proc[sp].iloc[0] for sp in species_cols])
                def lv_dynamics_2d_quiver(t, x):
                    full_state = np.zeros(n_species)
                    full_state[i] = x[0]
                    full_state[j] = x[1]
                    for k in range(n_species):
                        if k != i and k != j:
                            full_state[k] = x0_vals[k]
                    dxdt_full = full_state * (mu_inputs + alpha_inputs @ full_state)
                    return [dxdt_full[i], dxdt_full[j]]

                # Create grid for quiver plot
                grid_x_range_list = [quiver_settings["grid_x_min"], quiver_settings["grid_x_max"]]
                grid_y_range_list = [quiver_settings["grid_y_min"], quiver_settings["grid_y_max"]]
                grid_res = quiver_settings["grid_resolution"]
                
                X = np.linspace(grid_x_range_list[0], grid_x_range_list[1], grid_res)
                Y = np.linspace(grid_y_range_list[0], grid_y_range_list[1], grid_res)
                X_grid, Y_grid = np.meshgrid(X, Y)
                
                # Calculate derivatives at each grid point
                DX = np.zeros_like(X_grid)
                DY = np.zeros_like(Y_grid)
                
                for xi_idx in range(grid_res):
                    for yi_idx in range(grid_res):
                        derivatives = lv_dynamics_2d_quiver(0, [X_grid[yi_idx, xi_idx], Y_grid[yi_idx, xi_idx]])
                        DX[yi_idx, xi_idx] = derivatives[0]
                        DY[yi_idx, xi_idx] = derivatives[1]
                
                # Create quiver plot
                fig_quiver = go.Figure()
                
                # Efficiently add vector field arrows
                magnitude = np.sqrt(DX**2 + DY**2)
                has_vectors = np.any(magnitude > 1e-9)
                max_mag = np.max(magnitude) if has_vectors else 0.0

                # Debug info
                st.info(f"🔍 **Debug**: Max magnitude = {max_mag:.6f}, Non-zero vectors = {np.sum(magnitude > 1e-9)}")

                if has_vectors and max_mag > 1e-9:
                    # Debug grid ranges first
                    st.info(f"🔍 **Grid Range Debug**: X = [{grid_x_range_list[0]}, {grid_x_range_list[1]}], Y = [{grid_y_range_list[0]}, {grid_y_range_list[1]}], Grid res = {grid_res}")
                    
                    # Determine a good scale factor based on grid size - MUCH LARGER!
                    cell_width = (grid_x_range_list[1] - grid_x_range_list[0]) / grid_res
                    cell_height = (grid_y_range_list[1] - grid_y_range_list[0]) / grid_res
                    scale_factor = 2.0 * min(cell_width, cell_height)  # Increased from 0.4 to 2.0
                    
                    # Safety check: if scale factor is too small, use a minimum based on the data range
                    if scale_factor < 1e-6:
                        # Calculate a reasonable scale based on the overall data range
                        data_range_x = grid_x_range_list[1] - grid_x_range_list[0]
                        data_range_y = grid_y_range_list[1] - grid_y_range_list[0]
                        
                        # Handle case where ranges are zero or very small
                        if data_range_x < 1e-6:
                            data_range_x = 1.0  # Default fallback
                        if data_range_y < 1e-6:
                            data_range_y = 1.0  # Default fallback
                            
                        scale_factor = 0.1 * min(data_range_x, data_range_y) / grid_res
                        st.warning(f"⚠️ Scale factor was too small, using fallback: {scale_factor:.6f}")
                        
                        # Additional warning for identical ranges
                        if grid_y_range_list[0] == grid_y_range_list[1]:
                            st.error(f"🚨 **PROBLEM FOUND**: Y range is [{grid_y_range_list[0]}, {grid_y_range_list[1]}] - min and max are identical! Please set different Y min and max values in the quiver settings.")
                        if grid_x_range_list[0] == grid_x_range_list[1]:
                            st.error(f"🚨 **PROBLEM FOUND**: X range is [{grid_x_range_list[0]}, {grid_x_range_list[1]}] - min and max are identical! Please set different X min and max values in the quiver settings.")

                    # Normalize vectors and apply scale
                    DX_norm = DX / max_mag * scale_factor
                    DY_norm = DY / max_mag * scale_factor
                    
                    # Debug scale information
                    st.info(f"🔍 **Scale Debug**: Cell width = {cell_width:.4f}, Cell height = {cell_height:.4f}, Scale factor = {scale_factor:.4f}, Max arrow length = {scale_factor:.4f}")
                    
                    # Prepare lists for all line segments and hover points
                    arrow_lines_x, arrow_lines_y = [], []
                    hover_points_x, hover_points_y, hover_texts = [], [], []
                    arrows_added = 0
                    
                    # Add arrows using an efficient method
                    for xi_idx in range(0, grid_res, 2):  # Skip every other arrow for clarity
                        for yi_idx in range(0, grid_res, 2):
                            if magnitude[yi_idx, xi_idx] > 1e-9:
                                x_pos, y_pos = X_grid[yi_idx, xi_idx], Y_grid[yi_idx, xi_idx]
                                dx, dy = DX_norm[yi_idx, xi_idx], DY_norm[yi_idx, xi_idx]
                                x_end, y_end = x_pos + dx, y_pos + dy
                                
                                # Add arrow line segment
                                arrow_lines_x.extend([x_pos, x_end, None])
                                arrow_lines_y.extend([y_pos, y_end, None])
                                
                                # Add arrowhead segments
                                if abs(dx) > 1e-10 or abs(dy) > 1e-10:  # Only add arrowheads for non-zero vectors
                                    arrow_length = 0.3 * np.sqrt(dx**2 + dy**2)
                                    arrow_angle = np.pi / 6
                                    arrow_dir = np.arctan2(dy, dx)
                                    
                                    ax1 = x_end - arrow_length * np.cos(arrow_dir - arrow_angle)
                                    ay1 = y_end - arrow_length * np.sin(arrow_dir - arrow_angle)
                                    ax2 = x_end - arrow_length * np.cos(arrow_dir + arrow_angle)
                                    ay2 = y_end - arrow_length * np.sin(arrow_dir + arrow_angle)
                                    
                                    arrow_lines_x.extend([x_end, ax1, None, x_end, ax2, None])
                                    arrow_lines_y.extend([y_end, ay1, None, y_end, ay2, None])

                                # Add hover point
                                hover_points_x.append(x_pos)
                                hover_points_y.append(y_pos)
                                hover_texts.append(f"Vector at ({x_pos:.2f}, {y_pos:.2f})<br>dx/dt = {DX[yi_idx, xi_idx]:.3f}<br>dy/dt = {DY[yi_idx, xi_idx]:.3f}")
                                arrows_added += 1

                    # Debug info about arrows
                    st.info(f"🔍 **Arrow Debug**: Arrows added = {arrows_added}, Line segments = {len(arrow_lines_x)}")
                    
                    # Add arrow lines trace - but only if we actually have lines to draw
                    if len(arrow_lines_x) > 0:
                        # Add debug info about actual coordinates
                        sample_coords = [(arrow_lines_x[i], arrow_lines_y[i]) for i in range(0, min(10, len(arrow_lines_x)), 3) if arrow_lines_x[i] is not None]
                        st.info(f"🔍 **Coordinate Debug**: Sample arrow coords = {sample_coords[:3]}")
                        
                        fig_quiver.add_trace(go.Scatter(
                            x=arrow_lines_x, y=arrow_lines_y,
                            mode='lines', line=dict(width=4, color='red'),  # Much thicker and red for visibility
                            showlegend=False, hoverinfo='none',
                            name='Vector Field'
                        ))
                    
                    # Add invisible markers for hover text
                    if len(hover_points_x) > 0:
                        fig_quiver.add_trace(go.Scatter(
                            x=hover_points_x, y=hover_points_y, text=hover_texts,
                            mode='markers', marker=dict(size=8, color='rgba(255,0,0,0.3)'),
                            showlegend=False, hoverinfo='text',
                            name='Hover Points'
                        ))
                else:
                    # If no significant vectors, show a message on the plot
                    st.warning("⚠️ No significant vector field detected. All derivatives are near zero.")
                    # Add a simple grid of points to show the domain
                    grid_points_x, grid_points_y = [], []
                    for xi_idx in range(0, grid_res, 3):
                        for yi_idx in range(0, grid_res, 3):
                            grid_points_x.append(X_grid[yi_idx, xi_idx])
                            grid_points_y.append(Y_grid[yi_idx, xi_idx])
                    
                    fig_quiver.add_trace(go.Scatter(
                        x=grid_points_x, y=grid_points_y,
                        mode='markers', marker=dict(size=3, color='lightgray'),
                        showlegend=False, hoverinfo='text',
                        hovertext="Grid point - no significant flow"
                    ))

                # Update layout for quiver plot with proper aspect ratio
                grid_x_len = quiver_settings["grid_x_max"] - quiver_settings["grid_x_min"]
                grid_y_len = quiver_settings["grid_y_max"] - quiver_settings["grid_y_min"]
                
                fig_quiver.update_layout(
                    title=f"Vector Field: {sp_i} vs {sp_j}<br><sub>Arrows show direction and magnitude of population change</sub>",
                    xaxis_title=f"{sp_i} Population",
                    yaxis_title=f"{sp_j} Population",
                    template="plotly_white",
                    height=600,
                    showlegend=False,
                    hovermode='closest',
                    xaxis_range=grid_x_range_list,
                    yaxis_range=grid_y_range_list
                )
                
                # Add grid with proper aspect ratio
                aspect_ratio = grid_x_len / grid_y_len if grid_y_len > 0 else 1.0
                base_width = 700
                plot_height = int(base_width / aspect_ratio) if aspect_ratio > 0 else 600
                plot_height = max(400, min(800, plot_height))
                
                fig_quiver.update_layout(width=base_width, height=plot_height)
                
                fig_quiver.update_xaxes(
                    showgrid=True, gridwidth=1, gridcolor='lightgray',
                    scaleanchor="y", scaleratio=1
                )
                fig_quiver.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                st.plotly_chart(fig_quiver, use_container_width=False)
                
                # Add interpretation help for quiver plot
                with st.expander("💡 Vector Field Interpretation Guide", expanded=False):
                    st.markdown("""
                    **How to read the vector field:**
                    - **Arrows**: Show direction of population change at each point
                    - **Arrow length**: Proportional to rate of change magnitude
                    - **Arrow direction**: Points toward the direction of population flow
                    - **Dense arrows**: Rapid population changes
                    - **Sparse arrows**: Slow population changes
                    
                    **Flow patterns:**
                    - **Convergent arrows**: Stable equilibrium points
                    - **Divergent arrows**: Unstable equilibrium points  
                    - **Circular patterns**: Oscillatory dynamics
                    - **Parallel arrows**: Linear growth/decline
                    """)
                
                st.markdown(get_svg_download_link(fig_quiver, f"vector_field_{sp_i}_vs_{sp_j}"), unsafe_allow_html=True)

            # === NETWORK VISUALIZATION (SEPARATE SECTION) ===
            st.markdown("### 🕸️ Species Interaction Network")
            
            # Check if we have LV fit results for network visualization
            if "A_fit" in st.session_state and st.session_state.A_fit is not None:
                A_matrix = st.session_state.A_fit
                
                # Add debugging info to help users understand the matrix
                with st.expander("🔍 Interaction Matrix Debug Info", expanded=False):
                    st.markdown("**Raw Interaction Matrix (A):**")
                    
                    # Format the dataframe with higher precision
                    formatted_matrix_df = pd.DataFrame(A_matrix, index=species_cols, columns=species_cols)
                    formatted_matrix_df = formatted_matrix_df.round(8)  # Round to 8 decimal places
                    st.dataframe(formatted_matrix_df, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Matrix Max Value", f"{np.max(np.abs(A_matrix)):.8f}")
                    with col2:
                        st.metric("Matrix Min Value", f"{np.min(np.abs(A_matrix[A_matrix != 0])):.8f}" if np.any(A_matrix != 0) else "0")
                    with col3:
                        st.metric("Non-zero Elements", f"{np.count_nonzero(A_matrix)}")
                    
                    st.markdown("💡 **Tip**: If you see 0 interactions below, try lowering the 'Min Edge Threshold' to include weaker interactions.")
                
                # Network configuration UI  
                with st.form("network_visualization_form"):
                    st.markdown("**⚙️ Network Visualization Settings**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📍 Layout & Nodes**")
                        layout_type = st.selectbox(
                            "Layout Algorithm",
                            ["spring", "circular", "grid", "random"],
                            index=0,
                            help="How to arrange species nodes"
                        )
                        
                        node_size_factor = st.slider(
                            "Node Size Factor", 
                            min_value=20, max_value=100, value=50,
                            help="Base size of species nodes"
                        )
                        
                        scale_nodes_by_mu = st.checkbox(
                            "Scale Nodes by Growth Rate (μ)",
                            value=True,
                            help="Make nodes larger for species with higher growth rates"
                        )
                        
                    with col2:
                        st.markdown("**🔗 Edge Scaling**")
                        min_edge_threshold = st.slider(
                            "Min Edge Threshold",
                            min_value=0.00000001, max_value=1.0, value=0.00000001, step=0.001,
                            help="Hide interactions weaker than this threshold"
                        )
                        
                        # Smart threshold adjustment
                        if A_matrix is not None:
                            max_abs_val = np.max(np.abs(A_matrix))
                            if max_abs_val < min_edge_threshold and max_abs_val > 0:
                                suggested_threshold = max_abs_val * 0.1  # 10% of max value
                                st.warning(f"⚠️ Current threshold ({min_edge_threshold:.3f}) filters out all interactions! "
                                         f"Maximum interaction strength is {max_abs_val:.6f}. "
                                         f"Consider lowering threshold to ~{suggested_threshold:.6f}")
                        
                        edge_width_factor = st.slider(
                            "Edge Width Factor",
                            min_value=1, max_value=50, value=15,
                            help="Scale factor for edge thickness (proportional to interaction strength)"
                        )
                        
                        scale_edges_nonlinear = st.checkbox(
                            "Non-linear Edge Scaling",
                            value=True,
                            help="Use square root scaling for edge thickness (emphasizes strong interactions)"
                        )
                        
                    with col3:
                        st.markdown("**🎨 Visual Options**")
                        show_self_interactions = st.checkbox(
                            "Show Self-Interactions",
                            value=False,
                            help="Show diagonal elements (self-regulation)"
                        )
                        
                        color_scheme = st.selectbox(
                            "Color Scheme",
                            ["RdBu", "RdYlBu", "Spectral", "coolwarm", "seismic"],
                            index=0,
                            help="Color scheme for interaction strength"
                        )
                        
                        show_color_scale = st.checkbox(
                            "Show Color Scale",
                            value=True,
                            help="Display color scale legend"
                        )
                        
                        show_node_labels = st.checkbox(
                            "Show Node Labels",
                            value=True,
                            help="Display species names on nodes"
                        )
                    
                    # Submit button for network visualization settings
                    network_submit = st.form_submit_button("🔄 Update Network Plot")
                
                # Create network graph
                try:
                    import networkx as nx
                    
                    # Create directed graph
                    G = nx.DiGraph()
                    
                    # Add nodes (species)
                    for i, species in enumerate(species_cols):
                        G.add_node(i, species=species, mu=mu_inputs[i])
                    
                    # Add edges (interactions)
                    edge_weights = []
                    edge_colors = []
                    edge_info = []
                    
                    for i in range(n_species):
                        for j in range(n_species):
                            alpha_val = A_matrix[i, j]
                            
                            # Skip weak interactions or self-interactions if not wanted
                            if abs(alpha_val) < min_edge_threshold:
                                continue
                            if i == j and not show_self_interactions:
                                continue
                            
                            # Add edge from j to i (j affects i)
                            G.add_edge(j, i, weight=abs(alpha_val), alpha=alpha_val)
                            edge_weights.append(abs(alpha_val))
                            edge_colors.append(alpha_val)
                            edge_info.append(f"α({species_cols[i]},{species_cols[j]}) = {alpha_val:.8f}")
                    
                    # Generate layout with better algorithms
                    if layout_type == "circular":
                        pos = nx.circular_layout(G, scale=2)
                    elif layout_type == "spring":
                        # Use better spring layout parameters for more natural look
                        pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42, scale=2)
                    elif layout_type == "grid":
                        # Create a simple grid layout
                        n_cols = math.ceil(math.sqrt(n_species))
                        pos = {}
                        for i, node in enumerate(G.nodes()):
                            row = i // n_cols
                            col = i % n_cols
                            pos[node] = (col * 1.5, -row * 1.5)  # Increase spacing
                    else:  # random
                        pos = nx.random_layout(G, seed=42)
                        # Scale up random layout for better spacing
                        pos = {node: (coord[0] * 3, coord[1] * 3) for node, coord in pos.items()}
                    
                    # Create Plotly network visualization
                    fig_network = go.Figure()
                    
                    # Calculate scaling factors
                    if edge_colors:
                        max_abs_interaction = max(abs(c) for c in edge_colors)
                    else:
                        max_abs_interaction = 1.0
                    
                    # Calculate node sizes
                    if scale_nodes_by_mu:
                        mu_values = [mu_inputs[i] for i in range(n_species)]
                        max_mu = max(mu_values) if mu_values else 1.0
                        min_mu = min(mu_values) if mu_values else 0.0
                        mu_range = max_mu - min_mu if max_mu > min_mu else 1.0
                        
                        node_sizes = []
                        for i in range(n_species):
                            # Scale node size between 0.5x and 2x the base size based on mu
                            mu_norm = (mu_inputs[i] - min_mu) / mu_range if mu_range > 0 else 0.5
                            scaled_size = node_size_factor * (0.5 + 1.5 * mu_norm)
                            node_sizes.append(scaled_size)
                    else:
                        node_sizes = [node_size_factor] * n_species
                    
                    # Add edges with enhanced scaling and self-loop support
                    for edge, weight, color, info in zip(G.edges(), edge_weights, edge_colors, edge_info):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        
                        # Determine edge color based on interaction type
                        if color_scheme == "RdBu":
                            if color > 0:
                                edge_color = f'rgba(0, 100, 255, 0.8)'  # Blue for positive
                            else:
                                edge_color = f'rgba(255, 50, 50, 0.8)'  # Red for negative
                        else:
                            # Default red/blue
                            edge_color = f'rgba({255 if color < 0 else 0}, 0, {255 if color > 0 else 0}, 0.8)'
                        
                        # Calculate scaled edge width
                        if scale_edges_nonlinear:
                            scaled_width = edge_width_factor * (weight ** 0.5) / (max_abs_interaction ** 0.5) if max_abs_interaction > 0 else 1
                        else:
                            scaled_width = edge_width_factor * weight / max_abs_interaction if max_abs_interaction > 0 else 1
                        
                        # Handle self-loops differently from regular edges
                        if edge[0] == edge[1]:  # Self-loop
                            # Create a circular self-loop
                            node_radius = node_sizes[edge[0]] * 0.003  # Node radius estimate
                            loop_radius = max(0.3, node_radius * 3)  # Loop radius
                            
                            # Create circle points for self-loop
                            theta = np.linspace(0, 2*np.pi, 50)
                            loop_x = x0 + loop_radius + loop_radius * np.cos(theta)
                            loop_y = y0 + loop_radius * np.sin(theta)
                            
                            # Add self-loop circle
                            fig_network.add_trace(go.Scatter(
                                x=loop_x,
                                y=loop_y,
                                mode='lines',
                                line=dict(
                                    color=edge_color,
                                    width=max(2, scaled_width)
                                ),
                                showlegend=False,
                                hoverinfo='text',
                                hovertext=f"Self-interaction: {info}",
                                name=f"Self-loop {edge[0]}"
                            ))
                            
                            # Add arrow for self-loop
                            arrow_pos = len(theta) // 4  # Position arrow at 90 degrees
                            tip_x = loop_x[arrow_pos]
                            tip_y = loop_y[arrow_pos]
                            
                            # Calculate tangent direction at arrow position
                            if arrow_pos < len(theta) - 1:
                                dx_tangent = loop_x[arrow_pos + 1] - loop_x[arrow_pos]
                                dy_tangent = loop_y[arrow_pos + 1] - loop_y[arrow_pos]
                                tangent_length = np.sqrt(dx_tangent**2 + dy_tangent**2)
                                if tangent_length > 0:
                                    dx_tangent /= tangent_length
                                    dy_tangent /= tangent_length
                                else:
                                    dx_tangent, dy_tangent = 1, 0
                            else:
                                dx_tangent, dy_tangent = 1, 0
                            
                            # Create arrow for self-loop
                            arrow_size = 0.08  # Smaller self-loop arrow
                            arrow_angle = np.pi / 6  # 30 degrees for narrower arrow
                            
                            base_x1 = tip_x - arrow_size * np.cos(np.arctan2(dy_tangent, dx_tangent) - arrow_angle)
                            base_y1 = tip_y - arrow_size * np.sin(np.arctan2(dy_tangent, dx_tangent) - arrow_angle)
                            base_x2 = tip_x - arrow_size * np.cos(np.arctan2(dy_tangent, dx_tangent) + arrow_angle)
                            base_y2 = tip_y - arrow_size * np.sin(np.arctan2(dy_tangent, dx_tangent) + arrow_angle)
                            
                            # Add self-loop arrow
                            fig_network.add_trace(go.Scatter(
                                x=[tip_x, base_x1, base_x2, tip_x],
                                y=[tip_y, base_y1, base_y2, tip_y],
                                mode='lines',
                                fill='toself',
                                fillcolor=edge_color,
                                line=dict(color=edge_color, width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                        else:  # Regular edge (not self-loop)
                            # Add regular edge line
                            fig_network.add_trace(go.Scatter(
                                x=[x0, x1], y=[y0, y1],
                                mode='lines',
                                line=dict(
                                    color=edge_color,
                                    width=max(2, scaled_width)  # Increased minimum width
                                ),
                                showlegend=False,
                                hoverinfo='text',
                                hovertext=info
                            ))
                            
                            # Add arrowhead for regular edges
                            # Calculate arrow direction
                            dx, dy = x1 - x0, y1 - y0
                            length = (dx**2 + dy**2)**0.5
                            if length > 0:
                                # Normalize direction
                                dx_norm, dy_norm = dx/length, dy/length
                                
                                # Smaller arrow size for better proportions
                                arrow_size = max(0.05, length * 0.05)  # Much smaller arrows
                                
                                # Position arrow closer to target node (leave space for node)
                                node_radius = node_sizes[edge[1]] * 0.003  # Estimate node radius
                                arrow_offset = node_radius + 0.03  # Smaller offset from node edge
                                
                                # Arrow tip position (slightly offset from target node)
                                tip_x = x1 - dx_norm * arrow_offset
                                tip_y = y1 - dy_norm * arrow_offset
                                
                                # Create a more compact triangular arrowhead
                                arrow_angle = np.pi / 6  # 30 degrees for narrower, less intrusive arrow
                                
                                # Calculate arrowhead base points
                                base_x1 = tip_x - arrow_size * np.cos(np.arctan2(dy_norm, dx_norm) - arrow_angle)
                                base_y1 = tip_y - arrow_size * np.sin(np.arctan2(dy_norm, dx_norm) - arrow_angle)
                                base_x2 = tip_x - arrow_size * np.cos(np.arctan2(dy_norm, dx_norm) + arrow_angle)
                                base_y2 = tip_y - arrow_size * np.sin(np.arctan2(dy_norm, dx_norm) + arrow_angle)
                                
                                # Create filled triangle arrowhead
                                arrow_x = [tip_x, base_x1, base_x2, tip_x]
                                arrow_y = [tip_y, base_y1, base_y2, tip_y]
                                
                                # Add arrowhead as filled triangle
                                fig_network.add_trace(go.Scatter(
                                    x=arrow_x, 
                                    y=arrow_y,
                                    mode='lines',
                                    fill='toself',
                                    fillcolor=edge_color,
                                    line=dict(color=edge_color, width=2),  # Reduced line width
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                # Adjust main edge line to stop before the arrow
                                edge_end_x = x1 - dx_norm * (arrow_offset + arrow_size * 0.5)  # Adjusted spacing
                                edge_end_y = y1 - dy_norm * (arrow_offset + arrow_size * 0.5)
                                
                                # Update the main edge line to stop before arrow
                                fig_network.data[-2].update(
                                    x=[x0, edge_end_x],
                                    y=[y0, edge_end_y]
                                )
                    
                    # Add nodes
                    node_x = [pos[node][0] for node in G.nodes()]
                    node_y = [pos[node][1] for node in G.nodes()]
                    node_text = [f"{species_cols[node]}<br>μ = {mu_inputs[node]:.4f}" for node in G.nodes()]
                    
                    fig_network.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text' if show_node_labels else 'markers',
                        marker=dict(
                            size=node_sizes,
                            color='lightblue',
                            line=dict(width=2, color='darkblue')
                        ),
                        text=[species_cols[node] for node in G.nodes()] if show_node_labels else [],
                        textposition="middle center",
                        textfont=dict(size=max(8, min(16, node_size_factor // 4)), color='black'),
                        hoverinfo='text',
                        hovertext=node_text,
                        showlegend=False
                    ))
                    
                    # Calculate proper aspect ratio to prevent stretching
                    # Get the range of node positions
                    x_positions = [pos[node][0] for node in G.nodes()]
                    y_positions = [pos[node][1] for node in G.nodes()]
                    
                    x_range = max(x_positions) - min(x_positions)
                    y_range = max(y_positions) - min(y_positions)
                    
                    # Add padding to the ranges to prevent arrow cutoff
                    padding = 0.8  # Increased padding to accommodate arrows
                    x_range_padded = x_range + 2 * padding
                    y_range_padded = y_range + 2 * padding
                    
                    # Calculate aspect ratio and set appropriate width/height
                    aspect_ratio = x_range_padded / y_range_padded if y_range_padded > 0 else 1.0
                    
                    # Set base height and calculate width to maintain aspect ratio
                    base_height = 600
                    calculated_width = int(base_height * aspect_ratio)
                    
                    # Limit width to reasonable bounds
                    final_width = max(400, min(1200, calculated_width))
                    final_height = int(final_width / aspect_ratio) if aspect_ratio > 0 else base_height
                    
                    # Update layout with proper dimensions and aspect ratio
                    fig_network.update_layout(
                        title=f"Species Interaction Network ({len(G.edges())} interactions)",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="Arrow direction: Source → Target<br>Width ∝ Interaction strength<br>Color: Red (negative) / Blue (positive)",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(size=10)
                        )],
                        xaxis=dict(
                            showgrid=False, 
                            zeroline=False, 
                            showticklabels=False,
                            range=[min(x_positions) - padding, max(x_positions) + padding],
                            scaleanchor="y",  # Lock aspect ratio
                            scaleratio=1      # 1:1 aspect ratio
                        ),
                        yaxis=dict(
                            showgrid=False, 
                            zeroline=False, 
                            showticklabels=False,
                            range=[min(y_positions) - padding, max(y_positions) + padding]
                        ),
                        plot_bgcolor='white',
                        width=final_width,
                        height=final_height
                    )
                    
                    # Display with fixed size to prevent stretching
                    # Use two columns: one for plot, one for statistics
                    col_plot, col_stats = st.columns([2, 1])  # 2:1 ratio for plot:stats
                    
                    with col_plot:
                        st.plotly_chart(fig_network, use_container_width=False)
                    
                    with col_stats:
                        # Network statistics (moved to sidebar column)
                        if 'edge_weights' in locals() and 'edge_colors' in locals():
                            st.markdown("#### 📊 Network Statistics")
                            
                            # Statistics in a more compact format
                            n_edges = len([w for w in edge_weights if w >= min_edge_threshold])
                            n_positive = len([c for c in edge_colors if c > 0])
                            n_negative = len([c for c in edge_colors if c < 0])
                            avg_strength = np.mean([abs(c) for c in edge_colors]) if edge_colors else 0
                            
                            st.metric("Total Interactions", n_edges)
                            st.metric("Positive", n_positive)
                            st.metric("Negative", n_negative)
                            st.metric("Avg Strength", f"{avg_strength:.8f}")
                            
                            # Quick interaction summary
                            st.markdown("#### 🔍 Quick Analysis")
                            if n_edges > 0:
                                pos_pct = (n_positive / n_edges) * 100
                                neg_pct = (n_negative / n_edges) * 100
                                
                                st.write(f"**Interaction Balance:**")
                                st.write(f"• Positive: {pos_pct:.1f}%")
                                st.write(f"• Negative: {neg_pct:.1f}%")
                                
                                # Interpretation
                                if pos_pct > 60:
                                    st.success("🤝 **Cooperative system** - Many positive interactions")
                                elif neg_pct > 60:
                                    st.warning("⚔️ **Competitive system** - Many negative interactions")
                                else:
                                    st.info("⚖️ **Mixed system** - Balanced positive/negative interactions")
                                
                                # Network density
                                max_possible_edges = n_species * (n_species - 1)
                                if not show_self_interactions:
                                    actual_edges = n_edges
                                else:
                                    actual_edges = n_edges
                                    max_possible_edges += n_species  # Add self-loops
                                
                                density = (actual_edges / max_possible_edges) * 100 if max_possible_edges > 0 else 0
                                st.write(f"**Network Density:** {density:.1f}%")
                                
                                if density > 50:
                                    st.write("🕸️ **Dense network** - Many interactions")
                                elif density > 20:
                                    st.write("🔗 **Moderate connectivity**")
                                else:
                                    st.write("🌟 **Sparse network** - Few interactions")
                            
                            # Species ranking by influence
                            st.markdown("#### 🏆 Species Influence")
                            influence_scores = {}
                            for i, species in enumerate(species_cols):
                                # Calculate total outgoing influence (how much this species affects others)
                                outgoing = sum(abs(A_matrix[j, i]) for j in range(n_species) if abs(A_matrix[j, i]) >= min_edge_threshold)
                                # Calculate total incoming influence (how much others affect this species)
                                incoming = sum(abs(A_matrix[i, j]) for j in range(n_species) if abs(A_matrix[i, j]) >= min_edge_threshold)
                                
                                influence_scores[species] = {
                                    'outgoing': outgoing,
                                    'incoming': incoming,
                                    'total': outgoing + incoming
                                }
                            
                            # Sort by total influence
                            sorted_species = sorted(influence_scores.items(), key=lambda x: x[1]['total'], reverse=True)
                            
                            for i, (species, scores) in enumerate(sorted_species[:3]):  # Top 3
                                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                                st.write(f"{rank_emoji} **{species}**")
                                st.write(f"   • Total: {scores['total']:.8f}")
                                st.write(f"   • Out: {scores['outgoing']:.8f}, In: {scores['incoming']:.3f}")
                    
                except ImportError:
                    st.error("NetworkX library is required for network visualization. Please install it with: pip install networkx")
                except Exception as e:
                    st.error(f"Error creating network visualization: {str(e)}")
                
                # Detailed statistics and export (full width below the plot)
                if 'edge_weights' in locals() and 'edge_colors' in locals():
                    st.markdown("---")  # Separator line
                # Detailed statistics and export (full width below the plot)
                if 'edge_weights' in locals() and 'edge_colors' in locals():
                    st.markdown("---")  # Separator line
                    
                    # Detailed interaction matrix and export options
                    col_matrix, col_export = st.columns([3, 1])
                    
                    with col_matrix:
                        # Interaction matrix table
                        with st.expander("📋 Detailed Interaction Matrix", expanded=False):
                            # Create a formatted matrix display
                            matrix_df = pd.DataFrame(A_matrix, 
                                                   index=[f"→ {sp}" for sp in species_cols],
                                                   columns=[f"{sp} →" for sp in species_cols])
                            
                            # Color formatting function
                            def color_matrix(val):
                                if pd.isna(val):
                                    return ''
                                elif val > 0:
                                    return f'background-color: rgba(0, 100, 200, {min(1, abs(val)*10)})'
                                elif val < 0:
                                    return f'background-color: rgba(200, 0, 0, {min(1, abs(val)*10)})'
                                else:
                                    return 'background-color: rgba(128, 128, 128, 0.1)'
                            
                            styled_df = matrix_df.style.map(color_matrix).format("{:.8f}")
                            st.write("**Interaction Matrix (α) - Rows affect Columns**")
                            st.dataframe(styled_df, use_container_width=True)
                            
                            st.markdown("""
                            **Color Legend:**
                            - 🔵 **Blue**: Positive interactions (facilitation/mutualism)  
                            - 🔴 **Red**: Negative interactions (competition/inhibition)
                            - **Intensity**: Darker colors = stronger interactions
                            """)
                    
                    with col_export:
                        # Download network data
                        st.markdown("#### 💾 Export Options")
                        
                        # Create network summary data
                        network_data = []
                        for i, (edge, weight, color, info) in enumerate(zip(G.edges(), edge_weights, edge_colors, edge_info)):
                            source_species = species_cols[edge[0]]
                            target_species = species_cols[edge[1]]
                            interaction_type = "negative" if color < 0 else "positive"
                            
                            network_data.append({
                                "Source": source_species,
                                "Target": target_species,
                                "Interaction_Strength": color,
                                "Absolute_Strength": weight,
                                "Interaction_Type": interaction_type,
                                "Alpha_Parameter": f"α({target_species},{source_species})"
                            })
                        
                        network_df = pd.DataFrame(network_data)
                        
                        st.markdown(download_csv(network_df, "network_interactions"), unsafe_allow_html=True)
                        st.markdown(get_svg_download_link(fig_network, "species_network"), unsafe_allow_html=True)
            else:
                st.info("🎯 **Network visualization requires Lotka-Volterra fitting results.**")
                st.markdown("""
                **To generate the network plot:**
                1. **Upload data** with time series for multiple species
                2. **Run Lotka-Volterra fitting** to estimate interaction parameters (α matrix)
                3. **Return here** to visualize the species interaction network
                
                **The network will show:**
                - **Nodes**: Species (sized by growth rate μ)
                - **Edges**: Interactions (colored by effect type, sized by strength)
                - **Arrows**: Direction of influence between species
                """)
                
                # Help section
                with st.expander("ℹ️ How to Interpret the Network", expanded=False):
                    st.markdown("""
                    ### Network Interpretation Guide
                    
                    **🎯 Nodes (Circles)**: Each species in your system
                    - Size can be adjusted with the "Node Size" slider
                    - Hover to see species name and growth rate
                    
                    **➡️ Edges (Arrows)**: Interaction between species
                    - **Arrow direction**: From affecting species → to affected species
                    - **Edge thickness**: Proportional to interaction strength
                    - **Edge color**: 
                        - 🔴 **Red/Warm colors**: Negative interactions (competition, inhibition)
                        - 🔵 **Blue/Cool colors**: Positive interactions (facilitation, mutualism)
                    
                    **🔄 Self-loops**: Self-interaction (density-dependent effects)
                    - Usually negative (self-limitation)
                    - Can be hidden with the "Show Self-Interactions" toggle
                    
                    **📊 Layout Options**:
                    - **Circular**: Species arranged in a circle
                    - **Spring**: Force-directed layout (similar species cluster together)
                    - **Grid**: Regular grid pattern
                    - **Random**: Random positions
                    
                    **⚙️ Filtering**:
                    - Use "Min Edge Threshold" to hide weak interactions
                    - Adjust "Edge Width Scaling" to emphasize strong interactions
                    
                    **🎨 Colors**:
                    - Different color schemes available (RdBu recommended)
                    - Color scale shows interaction strength range
                    """)

# ========== About Tab ==========
with main_tabs[3]:
    st.markdown('<h2 class="section-header">About Cofit Dashboard</h2>', unsafe_allow_html=True)
    
    # Introduction Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h3 style="color: white; margin-top: 0;">🧬 Welcome to Cofit Dashboard</h3>
        <p style="font-size: 1.1em; margin-bottom: 0;">
            A comprehensive platform for analyzing microbial ecosystem dynamics using the Lotka-Volterra model. 
            Explore species interactions, visualize population dynamics, and uncover ecological patterns in your data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Application Overview
        st.markdown("### 🔬 Application Overview")
        
        with st.expander("📊 **What is the Lotka-Volterra Model?**", expanded=False):
            st.markdown("""
            The **Lotka-Volterra model** is a foundational mathematical framework in ecology that describes 
            the dynamics of interacting populations. Originally developed for predator-prey relationships, 
            it has been extended to model complex multi-species ecosystems.
            
            **Mathematical Foundation:**
            ```
            dXᵢ/dt = Xᵢ(μᵢ + Σⱼ αᵢⱼXⱼ)
            ```
            
            Where:
            - **Xᵢ**: Population of species i
            - **μᵢ**: Intrinsic growth rate of species i
            - **αᵢⱼ**: Interaction coefficient (how species j affects species i)
            
            **Interaction Types:**
            - **αᵢⱼ > 0**: Positive interaction (facilitation, mutualism)
            - **αᵢⱼ < 0**: Negative interaction (competition, inhibition)
            - **αᵢⱼ = 0**: No interaction
            """)
        
        # Features Overview
        st.markdown("### ⚡ Key Features")
        
        feature_tabs = st.tabs(["📈 Data Analysis", "🔧 Model Fitting", "🕸️ Network Visualization", "📊 Advanced Tools"])
        
        with feature_tabs[0]:
            st.markdown("""
            **Data Upload & Processing:**
            - Support for multiple file formats (CSV, Excel)
            - Automatic replicate averaging
            - Interactive data visualization
            - Quality control and validation
            
            **Time Series Analysis:**
            - Growth curve visualization
            - Statistical summaries
            - Data export capabilities
            """)
        
        with feature_tabs[1]:
            st.markdown("""
            **Automated Parameter Estimation:**
            - Exponential growth rate (μ) fitting
            - Interaction matrix (α) estimation
            - Multiple optimization algorithms
            - Statistical validation
            
            **Model Validation:**
            - Goodness-of-fit metrics
            - Residual analysis
            - Cross-validation options
            """)
        
        with feature_tabs[2]:
            st.markdown("""
            **Interactive Network Plots:**
            - Species interaction networks
            - Customizable layouts (spring, circular, grid)
            - Edge filtering and scaling
            - Export capabilities
            
            **Network Analysis:**
            - Interaction strength ranking
            - Network density calculations
            - Community structure analysis
            """)
        
        with feature_tabs[3]:
            st.markdown("""
            **Phase Plane Analysis:**
            - 2D phase portraits
            - Multiple initial conditions
            - Trajectory visualization
            
            **Vector Field Analysis:**
            - Quiver plots with customizable grids
            - Nullcline visualization
            - Equilibrium point analysis
            
            **Manual Parameter Exploration:**
            - Interactive parameter adjustment
            - Real-time plot updates
            - Pairwise simulation comparisons
            """)
        
        # User Guide
        st.markdown("### 📚 Quick Start Guide")
        
        # Create a nicely styled getting started guide using st.info and st.markdown
        st.info("🚀 **Getting Started in 4 Easy Steps**")
        
        st.markdown("**1. 📁 Data Upload**")
        st.markdown("Upload your time series data (CSV/Excel format). Include time points and species abundance measurements.")
        
        st.markdown("**2. 📊 Data Analysis**")
        st.markdown("Review your data with interactive plots, check for quality issues, and explore growth patterns.")
        
        st.markdown("**3. 🔧 Model Fitting**")
        st.markdown("Run automated parameter estimation to determine growth rates (μ) and interaction coefficients (α).")
        
        st.markdown("**4. 🕸️ Visualization**")
        st.markdown("Explore results with network plots, phase planes, and advanced analysis tools.")
        
        # Technical Information
        st.markdown("### 🔧 Technical Details")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            **Framework & Libraries:**
            - **Streamlit**: Web application framework
            - **Plotly**: Interactive visualization
            - **NetworkX**: Graph analysis
            - **SciPy**: Scientific computing
            - **NumPy/Pandas**: Data manipulation
            """)
        
        with tech_col2:
            st.markdown("""
            **Algorithms Used:**
            - **Exponential fitting**: Least squares regression
            - **ODE solving**: Runge-Kutta methods
            - **Optimization**: Powell, Nelder-Mead
            - **Network layout**: Force-directed algorithms
            """)
    
    with col2:
        # Feedback Section
        st.markdown("### 💬 Feedback & Support")
        
        # Supervisor's Google Form
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 8px; border: 2px solid #2196f3; margin-bottom: 1.5rem;">
            <h4 style="color: #1976d2; margin-top: 0;">📋 Official Feedback Form</h4>
            <p>Please share your experience and suggestions with our research team:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Google Form link button
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <a href="https://forms.gle/BbNU8KrcnHGMwbjJ6" target="_blank" 
               style="background: linear-gradient(45deg, #2196f3, #21cbf3); 
                      color: white; padding: 12px 24px; text-decoration: none; 
                      border-radius: 25px; font-weight: bold; font-size: 16px;
                      box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
                      transition: all 0.3s ease;">
                🚀 Open Feedback Form
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Feedback Section
        st.markdown("---")
        st.markdown("#### 🔥 Quick Feedback")
        
        with st.form("quick_feedback_form"):
            st.markdown("**Rate your experience:**")
            rating = st.select_slider(
                "Overall satisfaction",
                options=["😞 Poor", "😐 Fair", "🙂 Good", "😊 Very Good", "🤩 Excellent"],
                value="🙂 Good"
            )
            
            feedback_type = st.selectbox(
                "Feedback category",
                ["General", "Bug Report", "Feature Request", "Data Issues", "UI/UX", "Performance"]
            )
            
            quick_feedback = st.text_area(
                "Your feedback (optional):",
                placeholder="Share your thoughts, suggestions, or report issues...",
                height=100
            )
            
            if st.form_submit_button("✨ Submit Quick Feedback"):
                if quick_feedback.strip():
                    # Save feedback to CSV file
                    if save_feedback_to_csv(rating, feedback_type, quick_feedback):
                        st.success("✅ Thank you for your feedback! Your input has been saved and helps us improve the application.")
                        st.balloons()
                        
                        # Also store in session state for immediate display
                        if 'feedback_log' not in st.session_state:
                            st.session_state.feedback_log = []
                        
                        st.session_state.feedback_log.append({
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'rating': rating,
                            'category': feedback_type,
                            'feedback': quick_feedback
                        })
                    else:
                        st.error("❌ There was an issue saving your feedback. Please try again or use the official feedback form above.")
                else:
                    # Save rating-only feedback to CSV
                    if save_feedback_to_csv(rating, feedback_type, "Rating only - no text feedback"):
                        st.success("✅ Rating submitted and saved! Thank you for your input.")
                    else:
                        st.error("❌ There was an issue saving your rating. Please try again.")
        
        # Contact Information
        st.markdown("---")
        st.markdown("#### 📞 Contact & Support")
        
        # Use native Streamlit components with actual contact info
        st.info("📧 **Technical Support**")
        st.markdown("**Developer:** Srinath Laka")
        st.markdown("**Email:** srinathlaka1@gmail.com")
        st.markdown("For technical issues or questions about the application functionality.")
        
        st.info("🎓 **Research Inquiries**") 
        st.markdown("**Institution:** Institute of Microbiology, FSU Jena")
        st.markdown("**Project Type:** HiWi (Student Assistant) Project")
        st.markdown("For questions about the Lotka-Volterra model implementation or ecological applications, please use the official feedback form.")
        
        st.info("🐛 **Bug Reports**")
        st.markdown("Found a bug? Please describe the issue using the quick feedback form above or contact: srinathlaka1@gmail.com")
        
        # App Information
        st.markdown("---")
        st.markdown("#### ℹ️ App Information")
        
        info_metrics = st.columns(2)
        with info_metrics[0]:
            st.metric("Version", "Beta")
            st.metric("Updated", "Jan 2025")
        with info_metrics[1]:
            st.metric("Framework", "Streamlit")
            st.metric("Status", "Development")
        
        # Recent Feedback (if any)
        if 'feedback_log' in st.session_state and st.session_state.feedback_log:
            st.markdown("---")
            st.markdown("#### 📝 Recent Feedback")
            
            with st.expander("View feedback log", expanded=False):
                for i, feedback in enumerate(reversed(st.session_state.feedback_log[-3:])):  # Show last 3
                    st.markdown(f"""
                    **{feedback['timestamp']}** | {feedback['rating']} | *{feedback['category']}*
                    
                    > {feedback['feedback']}
                    
                    ---
                    """)
    
    # Footer
    st.markdown("---")
    
    footer_cols = st.columns([1, 2, 1])
    with footer_cols[1]:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p><strong>🧬 Cofit Dashboard</strong></p>
            <p>Advancing microbial ecology through mathematical modeling</p>
            <p style="font-size: 0.9em;">Developed by Srinath Laka • Institute of Microbiology, FSU Jena</p>
            <p style="font-size: 0.8em;">Built with Streamlit • HiWi Project 2025</p>
        </div>
        """, unsafe_allow_html=True)

