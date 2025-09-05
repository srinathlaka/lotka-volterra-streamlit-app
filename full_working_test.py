import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline
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
            df = pd.read_csv(file, header=None)
        elif file.name.endswith('.xls'):
            try:
                df = pd.read_excel(file, engine='xlrd', header=None)
            except ImportError:
                st.error("📋 **Excel .xls files require the 'xlrd' library.**")
                st.info("Please install it using: `pip install xlrd>=2.0.1` or upload a CSV/.xlsx file instead.")
                return None
        elif file.name.endswith('.xlsx'):
            try:
                df = pd.read_excel(file, engine='openpyxl', header=None)
            except ImportError:
                st.error("📋 **Excel .xlsx files require the 'openpyxl' library.**")
                st.info("Please install it using: `pip install openpyxl` or upload a CSV file instead.")
                return None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel (.xls/.xlsx) file.")
            return None
        
        # Handle NA values - replace common NA representations with proper NaN
        df = df.replace(['', ' ', 'NA', 'N/A', 'na', 'n/a', 'NULL', 'null', 'None', 'none', '#N/A', '#VALUE!', '#REF!'], np.nan)
        
        return df
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
            df = pd.read_csv(file, header=header)
        elif file.name.endswith('.xls'):
            try:
                df = pd.read_excel(file, engine='xlrd', header=header)
            except ImportError:
                st.error("📋 **Excel .xls files require the 'xlrd' library.**")
                st.info("Please install it using: `pip install xlrd>=2.0.1` or upload a CSV/.xlsx file instead.")
                return None
        elif file.name.endswith('.xlsx'):
            try:
                df = pd.read_excel(file, engine='openpyxl', header=header)
            except ImportError:
                st.error("📋 **Excel .xlsx files require the 'openpyxl' library.**")
                st.info("Please install it using: `pip install openpyxl` or upload a CSV file instead.")
                return None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel (.xls/.xlsx) file.")
            return None
        
        # Handle NA values - replace common NA representations with proper NaN
        df = df.replace(['', ' ', 'NA', 'N/A', 'na', 'n/a', 'NULL', 'null', 'None', 'none', '#N/A', '#VALUE!', '#REF!'], np.nan)
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        if "xlrd" in str(e).lower():
            st.info("💡 **Tip**: Try converting your Excel file to CSV format, which doesn't require additional libraries.")
        return None

def check_and_report_na_values(dfs, file_names=None):
    """
    Check for NA values in uploaded data and provide detailed report
    Returns: (has_na, na_report, na_summary)
    """
    if file_names is None:
        file_names = [f"File {i+1}" for i in range(len(dfs))]
    
    has_na = False
    na_report = []
    na_summary = {}
    
    for i, df in enumerate(dfs):
        if df is None:
            continue
            
        # Check for NA values
        na_mask = df.isna()
        na_count = na_mask.sum().sum()
        
        if na_count > 0:
            has_na = True
            total_cells = df.shape[0] * df.shape[1]
            na_percentage = (na_count / total_cells) * 100
            
            # Detailed analysis per column
            col_na_info = []
            for col in df.columns:
                col_na_count = na_mask[col].sum()
                if col_na_count > 0:
                    col_na_percentage = (col_na_count / len(df)) * 100
                    col_na_info.append({
                        'column': col,
                        'na_count': col_na_count,
                        'na_percentage': col_na_percentage,
                        'na_rows': df[na_mask[col]].index.tolist()[:5]  # First 5 rows with NA
                    })
            
            na_report.append({
                'file_name': file_names[i],
                'file_index': i,
                'total_na': na_count,
                'total_cells': total_cells,
                'na_percentage': na_percentage,
                'columns_with_na': col_na_info,
                'shape': df.shape
            })
            
            na_summary[file_names[i]] = {
                'na_count': na_count,
                'na_percentage': na_percentage
            }
    
    return has_na, na_report, na_summary

def handle_na_values(df, method='drop_rows', fill_value=0):
    """
    Handle NA values in dataframe according to specified method
    Methods:
    - 'drop_rows': Remove rows with any NA values
    - 'drop_columns': Remove columns with any NA values  
    - 'fill_zero': Fill NA values with 0
    - 'fill_mean': Fill NA values with column mean
    - 'fill_median': Fill NA values with column median
    - 'fill_forward': Forward fill (use previous value)
    - 'fill_backward': Backward fill (use next value)
    - 'fill_interpolate': Linear interpolation
    - 'fill_custom': Fill with custom value
    """
    if df is None:
        return None
    
    df_clean = df.copy()
    
    if method == 'drop_rows':
        df_clean = df_clean.dropna()
    elif method == 'drop_columns':
        df_clean = df_clean.dropna(axis=1)
    elif method == 'fill_zero':
        df_clean = df_clean.fillna(0)
    elif method == 'fill_mean':
        # Only fill numeric columns with mean
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif method == 'fill_median':
        # Only fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif method == 'fill_forward':
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'fill_backward':
        df_clean = df_clean.fillna(method='bfill')
    elif method == 'fill_interpolate':
        # Only interpolate numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate()
    elif method == 'fill_custom':
        df_clean = df_clean.fillna(fill_value)
    
    return df_clean

def get_first_valid_value_and_time(series, time_series):
    """
    Get the first valid (non-NA) value and its corresponding time point
    Returns: (first_valid_value, first_valid_time_index, valid_data_fraction)
    """
    valid_mask = ~series.isna()
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return None, None, 0.0
    
    first_valid_idx = valid_indices[0]
    first_valid_value = series.iloc[first_valid_idx]
    valid_fraction = len(valid_indices) / len(series)
    
    return first_valid_value, first_valid_idx, valid_fraction

def estimate_initial_conditions_smart(data_proc, species_cols, time_data):
    """
    Smart initial condition estimation for species with NA at start
    """
    initial_values = []
    initial_info = []
    
    for sp in species_cols:
        series = data_proc[sp]
        first_val, first_idx, valid_frac = get_first_valid_value_and_time(series, time_data)
        
        if first_val is None:
            # No valid data at all
            initial_values.append(1e-6)
            initial_info.append(f"{sp}: No valid data - using fallback (1e-6)")
        elif first_idx == 0:
            # Normal case - first timepoint is valid
            initial_values.append(first_val)
            initial_info.append(f"{sp}: Using actual t=0 value ({first_val:.4f})")
        else:
            # NA at start - extrapolate back to t=0
            if valid_frac < 0.3:
                # Too little valid data
                initial_values.append(first_val)
                initial_info.append(f"{sp}: Insufficient data ({valid_frac:.1%}) - using first valid ({first_val:.4f})")
            else:
                # Try to extrapolate back to t=0
                try:
                    # Get first few valid points for extrapolation
                    valid_data = series.dropna()
                    valid_mask = series.notna()
                    valid_times = time_data[valid_mask]
                    
                    if len(valid_data) >= 2:
                        # Linear extrapolation back to t=0
                        slope = (valid_data.iloc[1] - valid_data.iloc[0]) / (valid_times[1] - valid_times[0])
                        extrapolated = valid_data.iloc[0] - slope * (valid_times[0] - time_data[0])
                        
                        # Use extrapolated value if reasonable, otherwise first valid
                        if extrapolated > 0 and extrapolated < first_val * 10:
                            initial_values.append(extrapolated)
                            initial_info.append(f"{sp}: Extrapolated to t=0 ({extrapolated:.4f})")
                        else:
                            initial_values.append(first_val)
                            initial_info.append(f"{sp}: Extrapolation unreasonable - using first valid ({first_val:.4f})")
                    else:
                        initial_values.append(first_val)
                        initial_info.append(f"{sp}: Single point - using first valid ({first_val:.4f})")
                except Exception as e:
                    initial_values.append(first_val)
                    initial_info.append(f"{sp}: Extrapolation failed - using first valid ({first_val:.4f})")
    
    return np.array(initial_values), initial_info

def logistic_model(t, x0, mu, a11):
    """
    Modified growth model: dX/dt = μ(X + a11*X)
    Analytical solution: X(t) = X0 * exp(μ(1 + a11)(t - t0))
    """
    # Handle special cases to prevent numerical issues
    try:
        # Shift time to start from 0 for numerical stability
        t_shifted = t - t[0]
        
        # dX/dt = μ(X + a11*X) = μX(1 + a11)
        # Solution: X(t) = X0 * exp(μ(1 + a11)t)
        effective_rate = mu * (1 + a11)
        
        # Prevent overflow/underflow
        exponent = effective_rate * t_shifted
        exponent = np.clip(exponent, -700, 700)
        
        with np.errstate(over='ignore'):
            result = x0 * np.exp(exponent)
            
        # Ensure finite results
        result = np.where(np.isfinite(result), result, 
                         np.finfo(float).max if effective_rate > 0 else x0)
        
        # Ensure positive values
        result = np.maximum(result, 1e-15)
        
        return result
        
    except Exception:
        # Fallback to simple exponential with mu only
        with np.errstate(over='ignore'):
            exponent = mu * (t - t[0])
            exponent = np.clip(exponent, -700, 700)
            result = x0 * np.exp(exponent)
            return np.where(np.isfinite(result), result, x0)

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

# --- Missing data interpolation helper ---
def interpolate_missing_df(df, method="linear", fill_ends="nearest", spline_k=3, spline_s=0.0, floor_zero=True, min_clip=0.0, exclude_cols=None):
    """
    Fill NaNs per numeric column using time as x-axis.
    method: 'linear' | 'spline'
    fill_ends: 'nearest' (hold first/last) | 'extrapolate'
    spline_k: spline order (1..5)
    spline_s: smoothing factor (0=no smoothing)
    floor_zero: clip negatives to min_clip (default 0)
    min_clip: minimum value to clip to when floor_zero is True
    exclude_cols: set of column names to skip (e.g., {'background_avg'})
    Returns: (df_filled, total_filled_count, per_column_filled_dict)
    """
    if df is None or "time" not in df.columns:
        return df, 0, {}
    out = df.copy()
    t = np.asarray(out["time"].values, dtype=float)
    filled_total = 0
    per_col = {}
    if exclude_cols is None:
        exclude_cols = set()
    for col in out.columns:
        if col == "time" or col in exclude_cols:
            continue
        # numeric only
        try:
            y = pd.to_numeric(out[col], errors="coerce").astype(float).values
        except Exception:
            continue
        nan_count = int(np.isnan(y).sum())
        if nan_count == 0:
            continue
        mask = np.isfinite(y)
        if mask.sum() < 2:
            # cannot interpolate; simple nearest fill if at least 1 point exists, else leave as-is
            if mask.any():
                y_new = pd.Series(y).interpolate(method="nearest", limit_direction="both").values
                if floor_zero:
                    y_new = np.maximum(y_new, float(min_clip))
                out[col] = y_new
                filled_total += nan_count
                per_col[col] = nan_count
            else:
                # no finite data → leave NaNs in place
                per_col[col] = 0
            continue

        x_obs = t[mask]
        y_obs = y[mask]
        try:
            if method == "spline":
                k = int(np.clip(spline_k, 1, 5))
                k = min(k, mask.sum() - 1)
                spl = UnivariateSpline(x_obs, y_obs, k=k, s=float(spline_s))
                y_interp = spl(t)
                if fill_ends == "nearest":
                    # hold boundary observed values
                    left_val = y_obs[0]
                    right_val = y_obs[-1]
                    y_interp = np.where(t < x_obs[0], left_val, y_interp)
                    y_interp = np.where(t > x_obs[-1], right_val, y_interp)
            else:
                f = interp1d(
                    x_obs, y_obs,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(y_obs[0], y_obs[-1]) if fill_ends == "nearest" else "extrapolate",
                    assume_sorted=True
                )
                y_interp = f(t)
        except Exception:
            # fallback to simple nearest
            y_interp = pd.Series(y).interpolate(method="nearest", limit_direction="both").values

        if floor_zero:
            y_interp = np.maximum(y_interp, float(min_clip))

        # replace NaNs only
        y_out = np.where(np.isnan(y), y_interp, y)
        out[col] = y_out
        filled_total += nan_count
        per_col[col] = nan_count

    return out, filled_total, per_col

st.markdown('<h1 class="main-header">LV-Sim (Lotka-Volterra Simulator)</h1>', unsafe_allow_html=True)
main_tabs = st.tabs(["Data Upload", "Data Analysis", "Model Fitting", "About"])

# ========== Data Upload Tab ==========
with main_tabs[0]:
    st.markdown('<h2 class="section-header">Upload and Process Data</h2>', unsafe_allow_html=True)
    # ...existing code...

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
        # Store uploaded files and dataframes in session state for access across sections
        st.session_state.uploaded_files = uploaded_files
        st.session_state.uploaded_dfs = dfs
    # Proceed if files are uploaded now OR if we already have processed data from a previous run
    if (len(uploaded_files) == int(num_replicates) and int(num_replicates) > 0) or ('df_avg' in st.session_state):
        
        # === NA VALUE DETECTION ===
        # Check for NA values but DO NOT automatically convert them to 0
        # Use current uploads if available, otherwise fall back to stored uploads
        dfs_src = dfs if len(dfs) > 0 else st.session_state.get('uploaded_dfs', [])
        file_names = (
            [f.name for f in uploaded_files] if len(uploaded_files) > 0
            else [getattr(f, 'name', f'file_{i+1}') for i, f in enumerate(st.session_state.get('uploaded_files', []))]
        )
        if len(dfs_src) > 0:
            has_na, na_report, na_summary = check_and_report_na_values(dfs_src, file_names)
        else:
            has_na, na_report, na_summary = (False, None, {})
        
        if has_na:
            st.warning("⚠️ **Missing values detected in your data.**")
            # Show summary of what was found
            total_na_found = sum(summary['na_count'] for summary in na_summary.values())
            st.write(f"• **{total_na_found} missing values** found across all files.")
            st.info("💡 **Important**: Missing values will be preserved as NA and not converted to fake zero values. This maintains data integrity for proper model fitting.")
        
        # Do NOT set any automatic NA handling
        st.session_state.na_handling_method = None
        st.session_state.na_custom_fill_value = None
        
        # === EXPERIMENT CONFIGURATION SECTION ===
        st.subheader("📊 Experiment Configuration")
        st.markdown("**Configure your experiment details and data format before column mapping:**")
        
        with st.expander("ℹ️ Configuration Guide", expanded=True):
            st.markdown("""
            **Step A: Number of Species**
            - Set how many individual species are in your experiment
            - This determines expected pairwise combinations
            
            **Step B: Base Data Format**
            - **Format 1**: Pairwise total counts only (X1+X2, X1+X3, etc.)
            - **Format 2**: Individual species counts in focal species pairwise combinations
            - **Format 3**: Individual species counts in ALL pairwise combinations
            
            **Step C: Full Community Data**
            - **No full community**: Only mono-cultures and pairwise data
            - **Individual counts**: Each species counted separately in full community
            - **Total sum only**: All species summed together in full community
            
            **Step D: Focal Species (Format 2 only)**
            - Select which species is the focal species for detailed tracking
            """)

        # Create form for unified configuration
        with st.form("experiment_configuration_form"):
            # Row 1: Species Count
            st.markdown("**Step A: Number of Species**")
            species_count_from_form = st.number_input(
                "Number of species in your experiment:", 
                min_value=1, 
                max_value=20, 
                step=1,
                value=st.session_state.get('confirmed_species_count_val', 3),
                help="This determines how many individual species and pairwise combinations we expect in your data"
            )
            
            st.markdown("---")
            
            # Row 2: Format Selection (Two Columns)
            col_format, col_community = st.columns(2)
            
            with col_format:
                st.markdown("**Step B: Base Data Format**")
                base_format = st.selectbox(
                    "Select your base format:",
                    options=[
                        "format_1_total_pairs",
                        "format_2_focal_individual", 
                        "format_3_all_individual"
                    ],
                    format_func=lambda x: {
                        "format_1_total_pairs": "Format 1: Pairwise Total Counts",
                        "format_2_focal_individual": "Format 2: Focal Species Individual Counts", 
                        "format_3_all_individual": "Format 3: All Species Individual Counts"
                    }[x],
                    index=0,
                    help="Choose the format that best matches your experimental data structure"
                )
            
            with col_community:
                st.markdown("**Step C: Full Community Data**")
                community_data_type = st.selectbox(
                    "Full community data availability:",
                    options=[
                        "no_community",
                        "individual_counts_community",
                        "total_sum_community"
                    ],
                    format_func=lambda x: {
                        "no_community": "❌ No full community data",
                        "individual_counts_community": "✅ Individual species counts in community",
                        "total_sum_community": "✅ Total community sum only"
                    }[x],
                    index=0,
                    help="Choose what type of full community data your file contains"
                )
            
            # Row 3: Focal Species Selection (only show for Format 2)
            focal_species = None
            if base_format == "format_2_focal_individual":
                st.markdown("---")
                st.markdown("**Step D: Focal Species Selection (Format 2)**")
                
                if species_count_from_form > 1:
                    focal_species_options = [f"Species {i+1}" for i in range(species_count_from_form)]
                    focal_species = st.selectbox(
                        "Select focal species:",
                        options=focal_species_options,
                        index=0,
                        help="The focal species will be tracked individually in all its pairwise combinations"
                    )
                    
                    # Show preview of what will be tracked
                    focal_index = focal_species_options.index(focal_species)
                    other_species = [s for i, s in enumerate(focal_species_options) if i != focal_index]
                    
                    st.info(f"📋 **{focal_species} will be tracked in:**")
                    st.write("• Mono-culture")
                    for other in other_species:
                        st.write(f"• {focal_species} + {other} (both species counted separately)")
                    
                    if community_data_type == "individual_counts_community":
                        st.write("• Full community (individual species counts)")
                    elif community_data_type == "total_sum_community":
                        st.write("• Full community (total sum)")
                else:
                    st.warning("⚠️ Format 2 requires at least 2 species. Please increase your species count.")
            
            st.markdown("---")
            
            # Confirmation button
            config_confirmed = st.form_submit_button(
                "🔍 Confirm Configuration & Generate Column Mapping", 
                use_container_width=True,
                help="Click to finalize your configuration and generate the column mapping interface"
            )

        # Show configuration summary and handle confirmation
        if config_confirmed or 'experiment_config_confirmed' in st.session_state:
            if config_confirmed:
                # Store the confirmed configuration
                st.session_state.experiment_config_confirmed = {
                    'species_count': species_count_from_form,
                    'base_format': base_format,
                    'community_data_type': community_data_type,
                    'focal_species': focal_species,
                    'combined_format': f"{base_format}_{community_data_type}"
                }
                # Also store in the other format for compatibility
                st.session_state.confirmed_format_config = {
                    'species_count': species_count_from_form,
                    'base_format': base_format,
                    'community_data_type': community_data_type,
                    'focal_species': focal_species,
                    'combined_format': f"{base_format}_{community_data_type}"
                }
                st.session_state.confirmed_species_count_val = species_count_from_form
                st.success("✅ Configuration confirmed! Column mapping interface will be generated below.")
            
            # Get the confirmed configuration
            config = st.session_state.experiment_config_confirmed
            species_count_from_form = config['species_count']
            base_format = config['base_format']
            community_data_type = config['community_data_type']
            focal_species = config['focal_species']
            combined_format = config['combined_format']
            
            # Store values for later use
            st.session_state.selected_data_format = combined_format
            st.session_state.confirmed_species_count_val = species_count_from_form
            if focal_species:
                st.session_state.focal_species = focal_species
            
            # Show confirmed configuration summary
            st.info(f"""
            **✅ Confirmed Configuration:**
            • **Species Count**: {species_count_from_form}
            • **Base Format**: {base_format.replace('_', ' ').title()}
            • **Community Data**: {community_data_type.replace('_', ' ').title()}
            {f"• **Focal Species**: {focal_species}" if focal_species else ""}
            """)
            
            # === COLUMN MAPPING INTERFACE (Only appears after confirmation) ===
            st.subheader("📋 Column Mapping")
            st.markdown("**Map your data columns to the expected format:**")

        # Show format preview and data structure if configuration is confirmed
        if 'confirmed_format_config' in st.session_state:
            st.success("✅ Format configuration confirmed! Column mapping interface generated below.")
            
            # Get the confirmed configuration
            config = st.session_state.confirmed_format_config
            base_format = config['base_format']
            community_option = config['community_data_type']
            combined_format = config['combined_format']
            focal_species = config.get('focal_species', 'Species 1')
            
            # Store for later use
            st.session_state.selected_data_format = base_format  # Keep base format for compatibility
            st.session_state.community_data_option = community_option
            if focal_species:
                st.session_state.focal_species = focal_species
            
            # Show confirmed configuration summary
            with st.container():
                st.markdown("### 📋 Confirmed Configuration")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Base Format:** {base_format.replace('_', ' ').title()}")
                with col2:
                    st.info(f"**Community Data:** {community_option.replace('_', ' ').title()}")
                with col3:
                    if base_format == "format_2_focal_individual":
                        st.info(f"**Focal Species:** {focal_species}")
                    else:
                        st.info("**Focal Species:** N/A")
            
            # Show expected data structure based on confirmed configuration
            if species_count_from_form > 0:
                with st.expander("📊 Expected Data Structure for Your Configuration", expanded=True):
                    n_species = species_count_from_form
                    
                    # Generate expected column structure based on configuration
                    expected_columns = ["Time"]
                    
                    # Individual species mono-cultures (always present)
                    for i in range(n_species):
                        expected_columns.append(f"Species {i+1}")
                    
                    # Add format-specific columns
                    if base_format == "format_1_total_pairs":
                        # Pairwise total counts
                        for i in range(n_species):
                            for j in range(i+1, n_species):
                                expected_columns.append(f"Species {i+1}+{j+1}")
                    
                    elif base_format == "format_2_focal_individual":
                        # Individual counts in focal species pairwise
                        focal_index = int(focal_species.split()[-1]) - 1
                        for i in range(n_species):
                            if i != focal_index:
                                pair_indices = sorted([focal_index, i])
                                expected_columns.append(f"Species {pair_indices[0]+1}(Species {pair_indices[0]+1}+{pair_indices[1]+1})")
                                expected_columns.append(f"Species {pair_indices[1]+1}(Species {pair_indices[0]+1}+{pair_indices[1]+1})")
                    
                    elif base_format == "format_3_all_individual":
                        # Individual counts in all pairwise combinations
                        for i in range(n_species):
                            for j in range(i+1, n_species):
                                expected_columns.append(f"Species {i+1}(Species {i+1}+{j+1})")
                                expected_columns.append(f"Species {j+1}(Species {i+1}+{j+1})")
                    
                    # Add community data based on option
                    if community_option == "individual_counts":
                        for i in range(n_species):
                            expected_columns.append(f"Species {i+1}(Full Community)")
                    elif community_option == "total_sum_only":
                        expected_columns.append("Full Community Total")
                    
                    # Background (optional)
                    expected_columns.append("Background (optional)")
                    
                    st.markdown(f"**Expected columns for {n_species} species:**")
                    st.markdown(f"**Total expected columns: {len(expected_columns)}**")
                    
                    # Display in a nice format
                    for i, col in enumerate(expected_columns, 1):
                        if "optional" in col.lower():
                            st.write(f"{i}. {col} 🔶")
                        else:
                            st.write(f"{i}. {col}")
                    
                    st.markdown("🔶 = Optional column")
        
        # Step 3: Show Expected Data Structure (Legacy - only show if no confirmation yet)
        elif species_count_from_form > 0 and 'confirmed_format_config' not in st.session_state:
            st.info("👆 Please confirm your format selection above to see the expected data structure and generate column mapping interface.")
        
        # Only proceed to column mapping if format is confirmed
        if 'confirmed_format_config' in st.session_state:
            # Get the confirmed data format for column mapping
            config = st.session_state.confirmed_format_config
            selected_format = config['base_format']
            community_data_type = config.get('community_data_type', 'no_community')
            focal_species = config.get('focal_species', 'Species 1')
            
            if selected_format == "format_1_total_pairs":
                    # Calculate expected columns for Format 1
                    individual_species = [f"Species {i+1}" for i in range(species_count_from_form)]
                    pairwise_combinations = []
                    for i in range(species_count_from_form):
                        for j in range(i+1, species_count_from_form):
                            pairwise_combinations.append(f"Species {i+1}+{j+1}")
                    
                    total_combinations = len(pairwise_combinations)
                    total_expected_cols = 1 + species_count_from_form + total_combinations + 1  # Time + Species + Pairwise + Background
                    
                    st.markdown(f"""
                    **Format 1: Pairwise Total Counts**
                    
                    **For {species_count_from_form} species, your data should contain:**
                    
                    **Required columns:**
                    - **1 Time column**: Time points for your experiment
                    - **{species_count_from_form} Individual species columns**: {', '.join(individual_species)}
                    - **{total_combinations} Pairwise combination columns**: {', '.join(pairwise_combinations)}
                    - **1 Background column (optional)**: Background/blank measurements
                    
                    **Total expected columns: {total_expected_cols}** (including optional background)
                    **Total minimum columns: {total_expected_cols-1}** (excluding optional background)
                    """)
                
            elif selected_format == "format_2_focal_individual":
                    focal_species = st.session_state.get('focal_species', 'Species 1')
                    st.markdown(f"""
                    **Format 2: Individual Counts - Focal Species ({focal_species})**
                    
                    **For {species_count_from_form} species with {focal_species} as focal:**
                    
                    **Required columns:**
                    - **1 Time column**: Time points for your experiment
                    - **{species_count_from_form} Individual species mono-culture columns**: Species 1, Species 2, etc.
                    - **Individual counts in focal species pairwise**: For each focal species pair, separate counts for each species
                    - **Individual counts in full community**: Each species' count when all species are together
                    - **1 Background column (optional)**: Background/blank measurements
                    
                    **Example focal pairwise columns:**
                    """)
                    
                    # Show focal pairwise examples
                    focal_index = int(focal_species.split()[-1]) - 1
                    focal_examples = []
                    for i in range(species_count_from_form):
                        if i != focal_index:
                            focal_examples.append(f"{focal_species}({focal_species}+Species {i+1}), Species {i+1}({focal_species}+Species {i+1})")
                    
                    if focal_examples:
                        st.markdown("- " + "\n- ".join(focal_examples))
                    
                    st.markdown(f"""
                    **Full community columns:**
                    - Species 1(Full Community), Species 2(Full Community), ..., Species {species_count_from_form}(Full Community)
                    """)
                
            elif selected_format == "format_3_all_individual":
                    # Calculate expected columns for Format 3
                    n_pairwise = species_count_from_form * (species_count_from_form - 1) // 2
                    n_individual_in_pairs = n_pairwise * 2
                    total_expected_cols = 1 + species_count_from_form + n_individual_in_pairs + species_count_from_form + 1
                    
                    st.markdown(f"""
                    **Format 3: Individual Counts - All Combinations**
                    
                    **For {species_count_from_form} species, your data should contain:**
                    
                    **Required columns:**
                    - **1 Time column**: Time points for your experiment
                    - **{species_count_from_form} Individual species mono-culture columns**: Species 1, Species 2, etc.
                    - **{n_individual_in_pairs} Individual counts in ALL pairwise combinations**: Both species counted separately in each pair
                    - **{species_count_from_form} Individual counts in full community**: Each species' count when all species are together
                    - **1 Background column (optional)**: Background/blank measurements
                    
                    **Individual counts in pairwise (example for {species_count_from_form} species):**
                    """)
                    
                    # Show example pairwise individual columns
                    example_pairs = []
                    for i in range(species_count_from_form):
                        for j in range(i+1, species_count_from_form):
                            example_pairs.append(f"X{i+1}(X{i+1}+X{j+1}), X{j+1}(X{i+1}+X{j+1})")
                    
                    if len(example_pairs) <= 6:
                        st.markdown("- " + "\n- ".join(example_pairs))
                    else:
                        st.markdown("- " + "\n- ".join(example_pairs[:6]) + f"\n- ... and {len(example_pairs)-6} more pairwise combinations")
                    
                    st.markdown(f"""
                    **Individual counts in full community:**
                    - X1(X1+X2+...+X{species_count_from_form}), X2(X1+X2+...+X{species_count_from_form}), ..., X{species_count_from_form}(X1+X2+...+X{species_count_from_form})
                    
                    **Total expected columns: {total_expected_cols}** (including optional background)
                    """)
            
            if species_count_from_form > 4 and selected_format != "format_1_total_pairs":
                st.warning(f"⚠️ With {species_count_from_form} species in Format 2/3, you'll need many columns. Consider using Format 1 for simpler data management.")
            elif species_count_from_form > 5 and selected_format == "format_1_total_pairs":
                total_combinations = species_count_from_form * (species_count_from_form - 1) // 2
                st.warning(f"⚠️ With {species_count_from_form} species, you'll need {total_combinations} pairwise combinations. This grows quickly: n species requires n×(n-1)/2 pairwise combinations.")
            
            # Step 4: Column mapping interface (within confirmed configuration)
            st.subheader("Step 4: Map Your Data Columns")
            
            # Get column headers from the first uploaded file
            uploaded_dfs = st.session_state.get('uploaded_dfs', [])
            uploaded_files = st.session_state.get('uploaded_files', [])
            
            # Debug information
            if len(uploaded_dfs) == 0:
                st.warning("⚠️ No uploaded files found in session state. Please upload your data files in Step 1 above.")
                st.info("💡 Tip: Make sure to upload all replicate files and wait for the 'All files uploaded!' message before confirming your format configuration.")
            
            sample_df = uploaded_dfs[0] if uploaded_dfs else None
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
                            key="header_detection_checkbox",
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
                
                # Generate expected data types based on species count and selected format
                species_labels = [f"Species {i+1}" for i in range(species_count_from_form)]
                
                # Format-specific column generation based on confirmed configuration
                config = st.session_state.get('experiment_config_confirmed', {})
                base_format = config.get('base_format', selected_format)
                community_data_type = config.get('community_data_type', 'no_community')
                focal_species = config.get('focal_species', 'Species 1')
                
                if base_format == "format_1_total_pairs":
                    # Format 1: Pairwise total counts
                    pairwise_labels = []
                    for i in range(species_count_from_form):
                        for j in range(i+1, species_count_from_form):
                            pairwise_labels.append(f"Species {i+1} + Species {j+1}")
                    
                    # Add community data if specified
                    community_labels = []
                    if community_data_type == "individual_counts_community":
                        for i in range(species_count_from_form):
                            community_labels.append(f"Species {i+1} in Full Community")
                    elif community_data_type == "total_sum_community":
                        community_labels.append("Full Community Sum")
                    
                    additional_labels = pairwise_labels + community_labels
                    
                elif base_format == "format_2_focal_individual":
                    # Format 2: Individual counts in focal species pairwise + optional community
                    focal_index = int(focal_species.split()[-1]) - 1  # Extract index from "Species X"
                    
                    # Individual counts in focal species pairwise combinations
                    focal_pairwise_labels = []
                    for i in range(species_count_from_form):
                        if i != focal_index:
                            other_index = i
                            # Both species in the focal pairwise
                            focal_pairwise_labels.append(f"Species {focal_index+1} in (Species {focal_index+1} + Species {other_index+1})")
                            focal_pairwise_labels.append(f"Species {other_index+1} in (Species {focal_index+1} + Species {other_index+1})")
                    
                    # Add community data based on configuration
                    community_labels = []
                    if community_data_type == "individual_counts_community":
                        for i in range(species_count_from_form):
                            community_labels.append(f"Species {i+1} in Full Community")
                    elif community_data_type == "total_sum_community":
                        community_labels.append("Full Community Sum")
                    
                    additional_labels = focal_pairwise_labels + community_labels
                    
                elif base_format == "format_3_all_individual":
                    # Format 3: Individual counts in ALL pairwise + optional community
                    all_pairwise_labels = []
                    for i in range(species_count_from_form):
                        for j in range(i+1, species_count_from_form):
                            # Both species in each pairwise
                            all_pairwise_labels.append(f"Species {i+1} in (Species {i+1} + Species {j+1})")
                            all_pairwise_labels.append(f"Species {j+1} in (Species {i+1} + Species {j+1})")
                    
                    # Add community data based on configuration
                    community_labels = []
                    if community_data_type == "individual_counts_community":
                        for i in range(species_count_from_form):
                            community_labels.append(f"Species {i+1} in Full Community")
                    elif community_data_type == "total_sum_community":
                        community_labels.append("Full Community Sum")
                    
                    additional_labels = all_pairwise_labels + community_labels
            
                with st.form(key='column_mapping_form'):
                    st.markdown(f"**Map each required data type to a column from your file ({base_format.replace('_', ' ').title()}):**")
                
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
                    
                        # Individual species mono-culture columns
                        st.markdown("### 🧬 Individual Species Mono-cultures")
                        species_cols = st.columns(min(3, len(species_labels)))
                        for idx, species_label in enumerate(species_labels):
                            with species_cols[idx % len(species_cols)]:
                                default_idx = min(idx + 1, len(available_columns) - 1)  # Skip time column
                                selected = st.selectbox(
                                    f"**{species_label}** *",
                                    options=available_columns,
                                    index=default_idx,
                                    key=f"mapping_{species_label}",
                                    help=f"Required: Select column for {species_label} mono-culture data"
                                )
                                column_mapping[species_label] = selected
                    
                        # Format-specific additional columns
                        if base_format == "format_1_total_pairs":
                            st.markdown("### 🤝 Pairwise Co-cultures (Total Counts)")
                            if additional_labels:  # Only create columns if we have labels
                                pairwise_cols = st.columns(min(3, len(additional_labels)))
                                for idx, pairwise_label in enumerate(additional_labels):
                                    with pairwise_cols[idx % len(pairwise_cols)]:
                                        default_idx = min(idx + 1 + len(species_labels), len(available_columns) - 1)
                                        selected = st.selectbox(
                                            f"**{pairwise_label}** *",
                                            options=available_columns,
                                        index=default_idx,
                                        key=f"mapping_{pairwise_label}",
                                        help=f"Required: Select column for {pairwise_label} total co-culture count"
                                    )
                                    column_mapping[pairwise_label] = selected
                                    
                        elif base_format == "format_2_focal_individual":
                            st.markdown("### 🎯 Focal Species Pairwise (Individual Counts)")
                            focal_pairwise_count = len([l for l in additional_labels if "in (" in l and "Full Community" not in l])
                            if focal_pairwise_count > 0:  # Only create columns if we have focal pairwise data
                                focal_cols = st.columns(min(3, focal_pairwise_count))
                                focal_idx = 0
                                for label in additional_labels:
                                    if "in (" in label and "Full Community" not in label:
                                        with focal_cols[focal_idx % len(focal_cols)]:
                                            default_idx = min(focal_idx + 1 + len(species_labels), len(available_columns) - 1)
                                            selected = st.selectbox(
                                                f"**{label}** *",
                                                options=available_columns,
                                                index=default_idx,
                                                key=f"mapping_{label}",
                                                help=f"Required: Individual count of species in pairwise culture"
                                            )
                                        column_mapping[label] = selected
                                        focal_idx += 1
                                        
                            st.markdown("### 🌐 Full Community (Individual Counts)")
                            community_count = len([l for l in additional_labels if "Full Community" in l])
                            if community_count > 0:  # Only create columns if we have community data
                                community_cols = st.columns(min(3, community_count))
                                community_idx = 0
                                for label in additional_labels:
                                    if "Full Community" in label:
                                        with community_cols[community_idx % len(community_cols)]:
                                            default_idx = min(focal_pairwise_count + community_idx + 1 + len(species_labels), len(available_columns) - 1)
                                            selected = st.selectbox(
                                                f"**{label}** *",
                                                options=available_columns,
                                            index=default_idx,
                                            key=f"mapping_{label}",
                                            help=f"Required: Individual count of species in full community"
                                        )
                                        column_mapping[label] = selected
                                        community_idx += 1
                                        
                        elif base_format == "format_3_all_individual":
                            st.markdown("### 🤝 All Pairwise Combinations (Individual Counts)")
                            all_pairwise_count = len([l for l in additional_labels if "in (" in l and "Full Community" not in l])
                            if all_pairwise_count > 0:  # Only create columns if we have pairwise data
                                all_pairwise_cols = st.columns(min(4, max(1, all_pairwise_count)))
                                pairwise_idx = 0
                                for label in additional_labels:
                                    if "in (" in label and "Full Community" not in label:
                                        with all_pairwise_cols[pairwise_idx % len(all_pairwise_cols)]:
                                            default_idx = min(pairwise_idx + 1 + len(species_labels), len(available_columns) - 1)
                                            selected = st.selectbox(
                                                f"**{label}** *",
                                                options=available_columns,
                                                index=default_idx,
                                                key=f"mapping_{label}",
                                                help=f"Required: Individual count of species in pairwise culture"
                                            )
                                            column_mapping[label] = selected
                                            pairwise_idx += 1
                                        
                            st.markdown("### 🌐 Full Community (Individual Counts)")
                            community_count = len([l for l in additional_labels if "Full Community" in l])
                            if community_count > 0:  # Only create columns if we have community data
                                community_cols = st.columns(min(3, community_count))
                                community_idx = 0
                                for label in additional_labels:
                                    if "Full Community" in label:
                                        with community_cols[community_idx % len(community_cols)]:
                                            default_idx = min(all_pairwise_count + community_idx + 1 + len(species_labels), len(available_columns) - 1)
                                            selected = st.selectbox(
                                                f"**{label}** *",
                                                options=available_columns,
                                                index=default_idx,
                                                key=f"mapping_{label}",
                                                help=f"Required: Individual count of species in full community"
                                        )
                                        column_mapping[label] = selected
                                        community_idx += 1
                    
                        # Background column (optional for all formats)
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
                    
                        # Validation and preview
                        with st.expander("📋 Mapping Preview", expanded=True):
                            st.markdown(f"**Your column mapping ({base_format.replace('_', ' ').title()}):**")
                            for req_col, mapped_col in column_mapping.items():
                                st.write(f"• **{req_col}** → `{mapped_col}`")
                        
                            # Check for duplicate mappings
                            mapped_values = [v for v in column_mapping.values() if v != "None (skip this column)"]
                            duplicates = set([x for x in mapped_values if mapped_values.count(x) > 1])
                            if duplicates:
                                st.error(f"⚠️ Duplicate mappings detected: {', '.join(duplicates)}. Each column can only be mapped once.")
                    
                        # Additional settings
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            subtract_bg_form = st.checkbox(
                                "Subtract background from data?", 
                                value=False,
                                key="subtract_background_checkbox", # Added unique key
                                help="If enabled, background values will be subtracted from all other measurements"
                            )
                        with col2:
                            st.markdown("*Required fields")
                    
                        process_data_button = st.form_submit_button("Process Data with Column Mapping", use_container_width=True)

                # --- Data Processing Logic (moved outside the form) ---
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
                            
                                # Re-read all files with proper headers and apply NA handling
                                processed_dfs = []
                                na_method = st.session_state.get('na_handling_method', None)
                                na_fill_value = st.session_state.get('na_custom_fill_value', 0)
                            
                                # Use session state uploaded files
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
                                
                                    # Do NOT apply automatic NA conversion to 0
                                    # Preserve NA values for scientific integrity
                                    if na_method is not None:
                                        # Only handle specific string representations of missing values
                                        df = df.replace(['', ' ', 'NULL', 'null', 'None', 'none'], np.nan)
                                        # Do NOT use handle_na_values to avoid fake zero conversion
                                        # df = handle_na_values(df, na_method, na_fill_value)  # DISABLED
                                
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
                                
                                    # Average the data columns (format-aware processing)
                                    data_columns = [col for col in column_mapping.keys() if col != "Time"]
                                    averaged_data = {}
                                    averaged_data["time"] = time_data
                                
                                    # Create organized data structure for different column types
                                    data_structure = {
                                        'individual_species': {},  # mono-culture data
                                        'pairwise_combinations': {},  # pairwise co-culture data  
                                        'full_community': {}  # full community data
                                    }
                                    
                                    # Process species mono-culture columns (same for all formats)
                                    for i in range(species_count_from_form):
                                        species_label = f"Species {i+1}"
                                        if species_label in processed_dfs[0].columns:
                                            values = np.array([df[species_label].values for df in processed_dfs])
                                            new_name = f"x{i+1}"
                                            averaged_data[new_name] = np.mean(values, axis=0)
                                            data_structure['individual_species'][new_name] = {
                                                'original_label': species_label,
                                                'species_index': i+1
                                            }
                                
                                    # Format-specific processing
                                    if base_format == "format_1_total_pairs":
                                        # Process pairwise total count columns
                                        for i in range(species_count_from_form):
                                            for j in range(i+1, species_count_from_form):
                                                pairwise_label = f"Species {i+1} + Species {j+1}"
                                                if pairwise_label in processed_dfs[0].columns:
                                                    values = np.array([df[pairwise_label].values for df in processed_dfs])
                                                    new_name = f"x{i+1}+x{j+1}"
                                                    averaged_data[new_name] = np.mean(values, axis=0)
                                                    data_structure['pairwise_combinations'][new_name] = {
                                                        'original_label': pairwise_label,
                                                        'species_indices': [i+1, j+1],
                                                        'type': 'total_count'
                                                    }
                                        
                                        # Process full community data for Format 1
                                        if community_data_type == "individual_counts_community":
                                            for i in range(species_count_from_form):
                                                community_label = f"Species {i+1} in Full Community"
                                                if community_label in processed_dfs[0].columns:
                                                    values = np.array([df[community_label].values for df in processed_dfs])
                                                    new_name = f"x{i+1}_community"
                                                    averaged_data[new_name] = np.mean(values, axis=0)
                                                    data_structure['full_community'][new_name] = {
                                                        'original_label': community_label,
                                                        'species_index': i+1,
                                                        'type': 'individual_count_in_community'
                                                    }
                                        elif community_data_type == "total_sum_community":
                                            # Handle total community sum
                                            community_sum_label = "Full Community Sum"
                                            if community_sum_label in processed_dfs[0].columns:
                                                values = np.array([df[community_sum_label].values for df in processed_dfs])
                                                new_name = "total_community"
                                                averaged_data[new_name] = np.mean(values, axis=0)
                                                data_structure['full_community'][new_name] = {
                                                    'original_label': community_sum_label,
                                                    'type': 'total_community_sum'
                                                }
                                                    
                                    elif base_format == "format_2_focal_individual":
                                        # Process focal species individual pairwise data
                                        focal_species = st.session_state.get('focal_species', 'Species 1')
                                        focal_index = int(focal_species.split()[-1]) - 1
                                        
                                        # Keep individual pairwise data separate (not summed)
                                        for i in range(species_count_from_form):
                                            if i != focal_index:
                                                other_index = i
                                                
                                                # Get individual counts in pairwise
                                                focal_label = f"Species {focal_index+1} in (Species {focal_index+1} + Species {other_index+1})"
                                                other_label = f"Species {other_index+1} in (Species {focal_index+1} + Species {other_index+1})"
                                                
                                                if focal_label in processed_dfs[0].columns and other_label in processed_dfs[0].columns:
                                                    focal_values = np.array([df[focal_label].values for df in processed_dfs])
                                                    other_values = np.array([df[other_label].values for df in processed_dfs])
                                                    
                                                    # Store individual counts separately
                                                    min_idx, max_idx = sorted([focal_index, other_index])
                                                    focal_new_name = f"x{focal_index+1}(x{min_idx+1}+x{max_idx+1})"
                                                    other_new_name = f"x{other_index+1}(x{min_idx+1}+x{max_idx+1})"
                                                    
                                                    averaged_data[focal_new_name] = np.mean(focal_values, axis=0)
                                                    averaged_data[other_new_name] = np.mean(other_values, axis=0)
                                                    
                                                    data_structure['pairwise_combinations'][focal_new_name] = {
                                                        'original_label': focal_label,
                                                        'species_indices': [focal_index+1, other_index+1],
                                                        'individual_species': focal_index+1,
                                                        'type': 'individual_count_in_pair'
                                                    }
                                                    data_structure['pairwise_combinations'][other_new_name] = {
                                                        'original_label': other_label,
                                                        'species_indices': [focal_index+1, other_index+1],
                                                        'individual_species': other_index+1,
                                                        'type': 'individual_count_in_pair'
                                                    }
                                        
                                        # Process full community individual data for Format 2
                                        if community_data_type == "individual_counts_community":
                                            for i in range(species_count_from_form):
                                                community_label = f"Species {i+1} in Full Community"
                                                if community_label in processed_dfs[0].columns:
                                                    values = np.array([df[community_label].values for df in processed_dfs])
                                                    new_name = f"x{i+1}_community"
                                                    averaged_data[new_name] = np.mean(values, axis=0)
                                                    data_structure['full_community'][new_name] = {
                                                        'original_label': community_label,
                                                        'species_index': i+1,
                                                        'type': 'individual_count_in_community'
                                                    }
                                        elif community_data_type == "total_sum_community":
                                            # Handle total community sum
                                            community_sum_label = "Full Community Sum"
                                            if community_sum_label in processed_dfs[0].columns:
                                                values = np.array([df[community_sum_label].values for df in processed_dfs])
                                                new_name = "total_community"
                                                averaged_data[new_name] = np.mean(values, axis=0)
                                                data_structure['full_community'][new_name] = {
                                                    'original_label': community_sum_label,
                                                    'type': 'total_community_sum'
                                                }
                                                    
                                    elif base_format == "format_3_all_individual":
                                        # Process all individual pairwise data
                                        for i in range(species_count_from_form):
                                            for j in range(i+1, species_count_from_form):
                                                # Get individual counts in pairwise
                                                species1_label = f"Species {i+1} in (Species {i+1} + Species {j+1})"
                                                species2_label = f"Species {j+1} in (Species {i+1} + Species {j+1})"
                                                
                                                if species1_label in processed_dfs[0].columns and species2_label in processed_dfs[0].columns:
                                                    species1_values = np.array([df[species1_label].values for df in processed_dfs])
                                                    species2_values = np.array([df[species2_label].values for df in processed_dfs])
                                                    
                                                    # Store individual counts separately
                                                    species1_new_name = f"x{i+1}(x{i+1}+x{j+1})"
                                                    species2_new_name = f"x{j+1}(x{i+1}+x{j+1})"
                                                    
                                                    averaged_data[species1_new_name] = np.mean(species1_values, axis=0)
                                                    averaged_data[species2_new_name] = np.mean(species2_values, axis=0)
                                                    
                                                    data_structure['pairwise_combinations'][species1_new_name] = {
                                                        'original_label': species1_label,
                                                        'species_indices': [i+1, j+1],
                                                        'individual_species': i+1,
                                                        'type': 'individual_count_in_pair'
                                                    }
                                                    data_structure['pairwise_combinations'][species2_new_name] = {
                                                        'original_label': species2_label,
                                                        'species_indices': [i+1, j+1],
                                                        'individual_species': j+1,
                                                        'type': 'individual_count_in_pair'
                                                    }
                                                    
                                        # Process full community individual data for Format 3
                                        if community_data_type == "individual_counts_community":
                                            for i in range(species_count_from_form):
                                                community_label = f"Species {i+1} in Full Community"
                                                if community_label in processed_dfs[0].columns:
                                                    values = np.array([df[community_label].values for df in processed_dfs])
                                                    new_name = f"x{i+1}_community"
                                                    averaged_data[new_name] = np.mean(values, axis=0)
                                                    data_structure['full_community'][new_name] = {
                                                        'original_label': community_label,
                                                        'species_index': i+1,
                                                        'type': 'individual_count_in_community'
                                                    }
                                        elif community_data_type == "total_sum_community":
                                            # Handle total community sum
                                            community_sum_label = "Full Community Sum"
                                            if community_sum_label in processed_dfs[0].columns:
                                                values = np.array([df[community_sum_label].values for df in processed_dfs])
                                                new_name = "total_community"
                                                averaged_data[new_name] = np.mean(values, axis=0)
                                                data_structure['full_community'][new_name] = {
                                                    'original_label': community_sum_label,
                                                    'type': 'total_community_sum'
                                                }
                                
                                    # Process background if present (same for all formats)
                                    if "Background (optional)" in processed_dfs[0].columns:
                                        values = np.array([df["Background (optional)"].values for df in processed_dfs])
                                        averaged_data["background_avg"] = np.mean(values, axis=0)
                                    else:
                                        # If no background, add a column of zeros
                                        averaged_data["background_avg"] = np.zeros_like(time_data)
                                
                                    # Extract column lists from organized data structure
                                    species_cols = list(data_structure['individual_species'].keys())
                                    pairwise_cols = list(data_structure['pairwise_combinations'].keys())
                                    full_community_cols = list(data_structure['full_community'].keys())
                                
                                    # Create final dataframe
                                    final_columns = ["time"] + species_cols + pairwise_cols + full_community_cols + ["background_avg"]
                                    df_avg = pd.DataFrame({col: averaged_data[col] for col in final_columns if col in averaged_data})
                                
                                    # Final NA validation check
                                    final_na_count = df_avg.isna().sum().sum()
                                    if final_na_count > 0:
                                        st.warning(f"⚠️ **Final data check**: {final_na_count} NA values remain in the processed_dfs data.")
                                    else:
                                        if na_method is not None:
                                            st.success(f"✅ **All missing values automatically set to 0**. Data is ready for analysis.")
                                
                                    # Store in session state (including format information and organized data structure)
                                    st.session_state.df_avg = df_avg
                                    st.session_state.species_count = species_count_from_form
                                    st.session_state.subtract_bg = subtract_bg_form
                                    st.session_state.species_cols = species_cols
                                    st.session_state.pairwise_cols = pairwise_cols
                                    st.session_state.full_community_cols = full_community_cols
                                    st.session_state.data_structure = data_structure  # Store organized data structure
                                    st.session_state.column_mapping = column_mapping
                                    st.session_state.na_handling_applied = na_method
                                    st.session_state.data_format = base_format
                                
                                    format_name = base_format.replace('_', ' ').title()
                                    st.success(f"✅ Data processed successfully using {format_name}! Go to the Data Analysis section.")
                                    st.balloons()
                                
                                    # Show preview of processed data
                                    with st.expander("📊 Processed Data Preview", expanded=True):
                                        st.write(f"First 5 rows of your processed data ({format_name}):")
                                        st.dataframe(df_avg.head(), use_container_width=True)
                                        
                                        # Show organized data structure
                                        st.markdown("### 🗂️ Data Organization")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.markdown("**📈 Individual Species (Mono-cultures)**")
                                            for col_name, info in data_structure['individual_species'].items():
                                                st.write(f"• `{col_name}` - Species {info['species_index']}")
                                        
                                        with col2:
                                            st.markdown("**🤝 Pairwise Co-cultures**")
                                            for col_name, info in data_structure['pairwise_combinations'].items():
                                                if info['type'] == 'total_count':
                                                    st.write(f"• `{col_name}` - Total count")
                                                else:
                                                    st.write(f"• `{col_name}` - Species {info['individual_species']} in pair")
                                        
                                        with col3:
                                            st.markdown("**🌐 Full Community**")

                                            community_data_type = st.session_state.get('community_data_option', 'no_community')
                                            
                                            # Debug information
                                            st.write(f"Debug - Community Type: {community_data_type}")
                                            st.write(f"Debug - Full Community Keys: {list(data_structure.get('full_community', {}).keys())}")
                                            
                                            if community_data_type == "no_community":
                                                st.write("• No full community data configured")
                                            elif community_data_type == "individual_counts_community":
                                                if data_structure.get('full_community'):
                                                    for col_name, info in data_structure['full_community'].items():
                                                        st.write(f"• `{col_name}` - Species {info.get('species_index', '?')} in community")
                                                else:
                                                    st.write("• No full community data found in structure")
                                            elif community_data_type == "total_sum_community":
                                                if data_structure.get('full_community'):
                                                    for col_name, info in data_structure['full_community'].items():
                                                        st.write(f"• `{col_name}` - Total community biomass")
                                                else:
                                                    st.write("• No full community data found in structure")
                                            else:
                                                if data_structure.get('full_community'):
                                                    for col_name, info in data_structure['full_community'].items():
                                                        st.write(f"• `{col_name}` - {info.get('type', 'Unknown')}")
                                                else:
                                                    st.write(f"• Unknown community type: {community_data_type}")
                                                    st.write("• No full community data found in structure")
                                        
                                        # Show format-specific information
                                        if base_format == "format_2_focal_individual":
                                            st.info(f"ℹ️ **Format 2 Processing**: Individual counts preserved for focal species analysis. Focal species: {focal_species}")
                                        elif base_format == "format_3_all_individual":
                                            st.info(f"ℹ️ **Format 3 Processing**: Individual counts preserved for all pairwise combinations.")
                                        else:
                                            st.info(f"ℹ️ **Format 1 Processing**: Pairwise total counts used for standard Lotka-Volterra analysis.")

                                    # (Interpolation UI moved below, outside mapping flow, so it persists across reruns.)
                                
                            except Exception as e:
                                st.error(f"Error processing data: {str(e)}")
                                st.error("Please check your column mapping and file format.")
            else:
                st.warning("No data files detected. Please upload files first.")
        else:
            st.warning("No data files detected. Please upload files first.")

        # --- Fill Missing Values Section (only after mapping + background subtraction) ---
        if 'df_avg' in st.session_state and isinstance(st.session_state.get('df_avg'), pd.DataFrame):
            if 'confirmed_format_config' in st.session_state and st.session_state.get('subtract_bg', False):
                st.markdown("---")
                with st.expander("🧩 Fill Missing Values (optional)", expanded=st.session_state.get('interp_expanded', False)):
                    st.markdown("Choose how to handle missing values before fitting and plots. Applied to df_avg.")
                    with st.form("interp_form_upload", clear_on_submit=False):
                        method = st.selectbox(
                            "Missing data handling",
                            ["None (keep NA)", "Linear", "Spline"],
                            index=st.session_state.get('interp_method_index', 0),
                            help="Linear is robust; Spline can be smoother. None keeps gaps."
                        )
                        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                        with col_i1:
                            fill_ends = st.selectbox(
                                "End handling",
                                ["nearest", "extrapolate"],
                                index=st.session_state.get('interp_fill_ends_index', 0),
                                help="How to treat times before first/after last observation"
                            )
                        with col_i2:
                            floor_zero = st.checkbox(
                                "Clip negatives",
                                value=st.session_state.get('interp_floor_zero', True),
                                help="Keeps populations non-negative after interpolation"
                            )
                        with col_i3:
                            min_clip = st.number_input(
                                "Minimum clip value",
                                value=float(st.session_state.get('interp_min_clip', 1e-5)),
                                min_value=0.0,
                                step=1e-6,
                                format="%.6f"
                            )
                        with col_i4:
                            exclude_bg = st.checkbox(
                                "Exclude background",
                                value=st.session_state.get('interp_exclude_bg', True),
                                help="Do not alter background_avg"
                            )
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            spline_k = st.slider("Spline order k", 1, 5, st.session_state.get('interp_spline_k', 3), disabled=(method != "Spline"))
                        with col_s2:
                            spline_s = st.number_input("Spline smoothing s", value=float(st.session_state.get('interp_spline_s', 0.0)), step=0.1, disabled=(method != "Spline"))

                        submitted = st.form_submit_button("Apply interpolation", use_container_width=False)
                        if submitted:
                            st.session_state.interp_expanded = True
                            st.session_state.interp_method_index = ["None (keep NA)", "Linear", "Spline"].index(method)
                            st.session_state.interp_fill_ends_index = ["nearest", "extrapolate"].index(fill_ends)
                            st.session_state.interp_floor_zero = bool(floor_zero)
                            st.session_state.interp_exclude_bg = bool(exclude_bg)
                            st.session_state.interp_spline_k = int(spline_k)
                            st.session_state.interp_spline_s = float(spline_s)
                            st.session_state.interp_min_clip = float(min_clip)

                            if method == "None (keep NA)":
                                st.info("Kept NA values unchanged.")
                            else:
                                exclude_cols = {"background_avg"} if exclude_bg else set()
                                df_interp, n_filled, per_col = interpolate_missing_df(
                                    st.session_state.df_avg,
                                    method=("spline" if method == "Spline" else "linear"),
                                    fill_ends=fill_ends,
                                    spline_k=int(spline_k),
                                    spline_s=float(spline_s),
                                    floor_zero=bool(floor_zero),
                                    min_clip=float(min_clip),
                                    exclude_cols=exclude_cols
                                )
                                st.session_state.df_avg = df_interp
                                st.success(f"✅ Filled {n_filled} missing values across {sum(1 for v in per_col.values() if v>0)} columns.")
                                still_na = df_interp.columns[df_interp.isna().any()].tolist()
                                if still_na:
                                    st.warning("Some columns still contain NA (insufficient finite points). Left unchanged:")
                                    st.write(", ".join(still_na))
                                st.dataframe(df_interp.head(), use_container_width=True)
## (Removed global Fill Missing Values block; interpolation UI lives in Upload tab only)

# ========== Data Analysis Tab ==========
with main_tabs[1]:
    if 'df_avg' in st.session_state:
        st.markdown('<h2 class="section-header">Data Analysis</h2>', unsafe_allow_html=True)
        df_avg = st.session_state.df_avg
        species_count = st.session_state.species_count
        subtract_bg = st.session_state.subtract_bg
        species_cols = st.session_state.species_cols
        pairwise_cols = st.session_state.pairwise_cols
        data_tabs = st.tabs(["Data Table", "Individual Species", "Pairwise Co-cultures", "Full Community"])
        with data_tabs[0]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Averaged data from all replicates")
            with col2:
                st.markdown(download_csv(df_avg, "averaged_data"), unsafe_allow_html=True)
            
            # Show data quality information
            if 'na_handling_applied' in st.session_state and st.session_state.na_handling_applied:
                st.info(f"🔧 **Data preprocessing applied**: NA values were handled using '{st.session_state.na_handling_applied}' method.")
            
            # Check for any remaining NAs
            current_na_count = df_avg.isna().sum().sum()
            if current_na_count > 0:
                st.warning(f"⚠️ **Data quality note**: {current_na_count} NA values present in the final dataset.")
            else:
                st.success("✅ **Data quality**: No missing values in the dataset.")
            
            if subtract_bg:
                data_no_time = df_avg.iloc[:, 1:-1].subtract(df_avg["background_avg"], axis=0)
                st.info("Background subtraction was applied to this data.")
            else:
                data_no_time = df_avg.iloc[:, 1:-1]
            
            # Data view options
            view_option = st.radio("View data:", ["Raw data", "Processed data (after background subtraction)"])
            
            # Show full data table
            if view_option == "Raw data":
                st.dataframe(df_avg, use_container_width=True)
            else:
                display_df = pd.concat([df_avg[["time"]], data_no_time], axis=1)
                st.dataframe(display_df, use_container_width=True)
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
            
            # Get pairwise data from organized data structure
            data_structure = st.session_state.get('data_structure', {})
            pairwise_data = data_structure.get('pairwise_combinations', {})
            
            if pairwise_data:
                col1, col2 = st.columns([4, 1])
                with col2:
                    y_scale_pair = st.radio("Y-axis scale:", ["Linear", "Logarithmic"], key="pair_scale")
                    
                    # Show data type information based on format
                    data_format = st.session_state.get('data_format', 'format_1_total_pairs')
                    if data_format == 'format_1_total_pairs':
                        st.info("📊 **Format 1**: Showing pairwise total counts")
                    else:
                        st.info("📊 **Format 2/3**: Showing individual species counts within pairwise co-cultures")
                
                fig_pair = go.Figure()
                colors = px.colors.qualitative.Bold
                
                # Plot all pairwise columns using the organized data structure
                for idx, (col_name, col_info) in enumerate(pairwise_data.items()):
                    # Create a more descriptive name for the legend
                    if col_info['type'] == 'total_count':
                        display_name = f"{col_name} (Total)"
                    else:
                        species_idx = col_info.get('individual_species', '')
                        pair_info = f"Species {species_idx}" if species_idx else col_name
                        display_name = f"{col_name} ({pair_info})"
                    
                    fig_pair.add_trace(go.Scatter(
                        x=df_avg["time"],
                        y=data_no_time[col_name],
                        mode='lines',
                        name=display_name,
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
                
                # Use the actual pairwise column names for download
                pairwise_col_names = list(pairwise_data.keys())
                st.markdown(download_csv(pd.concat([df_avg[["time"]], data_no_time[pairwise_col_names]], axis=1), "pairwise_coculture_data"), unsafe_allow_html=True)
                st.markdown(get_svg_download_link(fig_pair, "pairwise_coculture_growth_curves"), unsafe_allow_html=True)
                
                # Show data organization info
                with st.expander("📋 Pairwise Data Details", expanded=False):
                    st.markdown("**Available pairwise measurements:**")
                    for col_name, col_info in pairwise_data.items():
                        if col_info['type'] == 'total_count':
                            st.write(f"• `{col_name}` - Total count of species {col_info['species_indices']}")
                        else:
                            st.write(f"• `{col_name}` - Species {col_info['individual_species']} count in pair {col_info['species_indices']}")
            else:
                st.warning("⚠️ No pairwise co-culture data available. Please check your data format and column mapping.")
        
        with data_tabs[3]:
            st.subheader("Full Community Measurements")
            
            # Get the data format and community data type
            data_format = st.session_state.get('data_format', 'format_1_total_pairs')
            community_data_type = st.session_state.get('community_data_option', 'no_community')
            
            if community_data_type == "no_community":
                # No full community data configured
                st.info("📊 **No Full Community Data Configured**")
                st.markdown("""
                **Your current configuration does not include full community data.**
                
                **To enable full community analysis:**
                1. Go back to **Data Upload** tab
                2. In Step C: Full Community Data, select either:
                   - **"Individual species counts in community"** - Track each species separately in full community
                   - **"Total community sum only"** - Track only total biomass of full community
                3. Re-configure and upload your data
                4. Return here for full community analysis
                
                **Available analysis with current data:**
                - Individual species mono-cultures
                - Pairwise co-cultures
                """)
                
            elif data_format == 'format_1_total_pairs' and community_data_type != "no_community":
                # Format 1 with community data (shouldn't happen but handle gracefully)
                st.warning("⚠️ **Configuration Mismatch**")
                st.markdown("""
                **Format 1 (Pairwise Total Counts)** typically does not include full community data.
                
                **Current configuration:**
                - Base Format: Format 1 (Pairwise total counts)
                - Community Data: {community_data_type}
                
                **Recommendation:** Use Format 2 or Format 3 for full community analysis.
                """.format(community_data_type=str(community_data_type).replace('_', ' ').title() if community_data_type is not None else 'No Community'))
                
            elif community_data_type == "individual_counts_community":
                # Individual species counts in full community
                full_community_cols = st.session_state.get('full_community_cols', [])
                data_structure = st.session_state.get('data_structure', {})
                
                if full_community_cols:
                    st.success(f"🧬 **Individual Community Counts**: Found {len(full_community_cols)} species tracked in full community")
                    
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        y_scale_community = st.radio("Y-axis scale:", ["Linear", "Logarithmic"], key="community_scale")
                        show_mono_comparison = st.checkbox("Compare with mono-cultures", value=True, 
                                                        help="Show individual species mono-cultures for comparison")
                        show_community_total = st.checkbox("Show community total", value=True,
                                                        help="Show sum of all individual counts in community")
                    
                    fig_community = go.Figure()
                    colors = px.colors.qualitative.Set1
                    colors_light = px.colors.qualitative.Pastel
                    
                    # Add individual species counts in full community
                    for idx, col in enumerate(full_community_cols):
                        fig_community.add_trace(go.Scatter(
                            x=df_avg["time"],
                            y=data_no_time[col],
                            mode='lines+markers',
                            name=f"{col} (in community)",
                            line=dict(color=colors[idx % len(colors)], width=3),
                            marker=dict(size=4)
                        ))
                    
                    # Add mono-culture comparison if requested
                    if show_mono_comparison and species_cols:
                        for idx, species_col in enumerate(species_cols):
                            fig_community.add_trace(go.Scatter(
                                x=df_avg["time"],
                                y=data_no_time[species_col],
                                mode='lines',
                                name=f"{species_col} (mono)",
                                line=dict(color=colors_light[idx % len(colors_light)], width=2, dash='dash'),
                                opacity=0.7
                            ))
                    
                    # Add total community if requested
                    if show_community_total:
                        total_community = np.sum([data_no_time[col] for col in full_community_cols], axis=0)
                        fig_community.add_trace(go.Scatter(
                            x=df_avg["time"],
                            y=total_community,
                            mode='lines',
                            name="Total Community",
                            line=dict(color='black', width=4)
                        ))
                    
                    fig_community.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Optical Density (OD)",
                        template="plotly_white",
                        height=500,
                        showlegend=True,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    if y_scale_community == "Logarithmic":
                        fig_community.update_yaxes(type="log")
                    st.plotly_chart(fig_community, use_container_width=True)
                    
                    # Download options
                    community_data_export = pd.concat([df_avg[["time"]], data_no_time[full_community_cols]], axis=1)
                    st.markdown(download_csv(community_data_export, "full_community_individual_data"), unsafe_allow_html=True)
                    st.markdown(get_svg_download_link(fig_community, "full_community_individual_growth_curves"), unsafe_allow_html=True)
                    
                    # Show data organization info
                    with st.expander("📋 Individual Community Data Details", expanded=False):
                        st.markdown("**Individual species measurements in full community:**")
                        community_data = data_structure.get('full_community', {})
                        for col_name, col_info in community_data.items():
                            st.write(f"• `{col_name}` - Species {col_info.get('species_index', '?')} count when all species are present")
                
                else:
                    st.warning("⚠️ **Individual Community Data Missing**: Expected for your configuration but not found.")
                    st.markdown("""
                    **Expected for your selected format:**
                    - Individual species counts within the full community co-culture
                    - These should be mapped during column mapping step
                    - Check that your data includes columns like "Species X in Full Community"
                    """)
                    
            elif community_data_type == "total_sum_community":
                # Total community sum only
                full_community_cols = st.session_state.get('full_community_cols', [])
                data_structure = st.session_state.get('data_structure', {})
                
                if full_community_cols:
                    # Should be only one column for total community
                    total_col = full_community_cols[0] if len(full_community_cols) == 1 else None
                    
                    if total_col:
                        st.success(f"📊 **Total Community Column**: Found `{total_col}`")
                        
                        col1, col2 = st.columns([4, 1])
                        with col2:
                            y_scale_community = st.radio("Y-axis scale:", ["Linear", "Logarithmic"], key="community_total_scale")
                            show_mono_comparison = st.checkbox("Compare with mono-cultures", value=True, 
                                                            help="Show individual species mono-cultures for comparison")
                            show_mono_sum = st.checkbox("Show sum of mono-cultures", value=True,
                                                      help="Show sum of all individual mono-cultures for comparison")
                        
                        fig_community = go.Figure()
                        colors = px.colors.qualitative.Set1
                        
                        # Add total community measurement
                        fig_community.add_trace(go.Scatter(
                            x=df_avg["time"],
                            y=data_no_time[total_col],
                            mode='lines+markers',
                            name="Total Community (observed)",
                            line=dict(color='blue', width=4),
                            marker=dict(size=6)
                        ))
                        
                        # Add mono-culture comparison if requested
                        if show_mono_comparison and species_cols:
                            for idx, species_col in enumerate(species_cols):
                                fig_community.add_trace(go.Scatter(
                                    x=df_avg["time"],
                                    y=data_no_time[species_col],
                                    mode='lines',
                                    name=f"{species_col} (mono)",
                                    line=dict(color=colors[idx % len(colors)], width=2, dash='dot'),
                                    opacity=0.7
                                ))
                        
                        # Add sum of mono-cultures if requested
                        if show_mono_sum and species_cols:
                            mono_sum = np.sum([data_no_time[col] for col in species_cols], axis=0)
                            fig_community.add_trace(go.Scatter(
                                x=df_avg["time"],
                                y=mono_sum,
                                mode='lines',
                                name="Sum of Mono-cultures",
                                line=dict(color='red', width=3, dash='dash')
                            ))
                        
                        fig_community.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Optical Density (OD)",
                            template="plotly_white",
                            height=500,
                            showlegend=True,
                            margin=dict(l=10, r=10, t=30, b=10)
                        )
                        if y_scale_community == "Logarithmic":
                            fig_community.update_yaxes(type="log")
                        st.plotly_chart(fig_community, use_container_width=True)
                        
                        # Download options
                        community_data_export = pd.concat([df_avg[["time"]], data_no_time[[total_col]]], axis=1)
                        st.markdown(download_csv(community_data_export, "full_community_total_data"), unsafe_allow_html=True)
                        st.markdown(get_svg_download_link(fig_community, "full_community_total_growth_curve"), unsafe_allow_html=True)
                        
                        # Show data organization info
                        with st.expander("📋 Total Community Data Details", expanded=False):
                            st.markdown("**Total community measurement:**")
                            community_data = data_structure.get('full_community', {})
                            for col_name, col_info in community_data.items():
                                st.write(f"• `{col_name}` - Total biomass of all species together")
                                
                        # Calculate basic statistics
                        with st.expander("📊 Community vs Mono-culture Statistics", expanded=False):
                            if species_cols:
                                mono_sum = np.sum([data_no_time[col] for col in species_cols], axis=0)
                                community_total = data_no_time[total_col]
                                
                                # Remove NaN values for comparison
                                valid_mask = ~(np.isnan(mono_sum) | np.isnan(community_total))
                                if valid_mask.sum() > 0:
                                    mono_valid = mono_sum[valid_mask]
                                    community_valid = community_total[valid_mask]
                                    
                                    # Calculate basic metrics
                                    mean_mono = np.mean(mono_valid)
                                    mean_community = np.mean(community_valid)
                                    ratio = mean_community / mean_mono if mean_mono > 0 else np.inf
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Mean Mono Sum", f"{mean_mono:.4f}")
                                    with col2:
                                        st.metric("Mean Community", f"{mean_community:.4f}")
                                    with col3:
                                        st.metric("Community/Mono Ratio", f"{ratio:.3f}")
                                    
                                    # Interpretation
                                    if ratio > 1.1:
                                        st.info("🔼 **Synergistic effect**: Community biomass > sum of mono-cultures")
                                    elif ratio < 0.9:
                                        st.warning("🔽 **Competitive effect**: Community biomass < sum of mono-cultures")
                                    else:
                                        st.success("➡️ **Additive effect**: Community ≈ sum of mono-cultures")
                    else:
                        st.error("⚠️ **Multiple total columns found**: Expected only one total community column")
                        st.write(f"Found columns: {full_community_cols}")
                        
                else:
                    st.warning("⚠️ **Total Community Data Missing**: Expected for your configuration but not found.")
                    st.markdown("""
                    **Expected for your selected format:**
                    - Single column containing total community biomass/population
                    - This should be mapped during column mapping step
                    - Check that your data includes a column like "Full Community Total"
                    """)
            
            else:
                st.error(f"⚠️ **Unknown community data type**: {community_data_type}")
                st.markdown("Please reconfigure your data upload with a valid community data option.")
    else:
        st.warning("Please upload and process data first!")

# ========== Model Fitting Tab ==========
with main_tabs[2]:
    if 'df_avg' in st.session_state:
        st.markdown('<h2 class="section-header">Model Fitting</h2>', unsafe_allow_html=True)
        
        # Add comprehensive explanation section
        with st.expander("📋 **Model Fitting Overview - What Each Tab Does**", expanded=True):
            st.markdown("""
            ### 🎯 **This Section Contains 6 Analysis Tabs**
            
            #### **🧪 Individual Species Analysis (Tabs 1-3):**
            **These analyze each species separately using mono-culture data**
            
            1. **📊 Parameter Table** - Shows fitted logistic parameters (x₀, μ, a₁₁) for each individual species
            2. **📈 Fitted Curves** - Visualizes how well the logistic model fits your individual species data  
            3. **⚖️ Growth Rate Comparison** - Compares growth rates (μ) between different species
            
            #### **🤝 Multi-Species Interaction Analysis (Tabs 4-6):**
            **These analyze species interactions using pairwise co-culture data**
            
            4. **🔬 LV Pairwise Theory** - Explains the Lotka-Volterra model for species interactions
            5. **🧮 LV Biological Fit** - **NEW!** Proper biological pairwise fitting methodology
            6. **🎨 LV Analysis & Plots** - Visualization tools and full community prediction
            
            ---
            
            ### 📈 **Recommended Workflow:**
            1. **Start with Individual Species** (Tabs 1-3) → Get basic growth parameters (μ, a₁₁)
            2. **Analyze Pairwise Interactions** (Tabs 4-5) → Estimate interaction matrix (α)
            3. **Validate & Explore** (Tab 6) → Full community prediction and phase plane analysis
            
            ### 🔬 **Data Requirements by Analysis Type:**
            - **Individual Species Analysis**: Mono-culture time series (X₁, X₂, X₃, X₄, X₅)
            - **Pairwise Interaction Analysis**: Individual species counts in pairwise cultures (X₁(X₁+X₂), X₂(X₁+X₂), etc.)
            - **Full Community Validation**: Individual species counts in full community (X₁(X₁+X₂+X₃+X₄+X₅), etc.)
            """)
        
        with st.expander("ℹ️ About the Single Culture Model", expanded=False):
            st.markdown("""
            For single culture fitting, the model used is:
            """)
            st.latex(r"\frac{dX_i}{dt} = \mu_i (X_i + a_{ii} X_i)")
            st.markdown("""
            where:
            - $\\mu_i$ is the growth rate of species $i$
            - $a_{ii}$ is the self-interaction coefficient  
            - $X_i$ is the population of species $i$
            
            Each species is fitted separately to obtain $\\mu_i$ and $a_{ii}$ parameters.
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
        
        # Better initial parameter estimation function
        def estimate_logistic_params_better(valid_time, valid_y):
            """
            Better initial parameter estimation for logistic model
            """
            # Ensure positive data
            valid_y = np.maximum(valid_y, 1e-8)
            
            # Initial value (first data point)
            x0_est = valid_y[0]
            
            # Estimate growth rate from early exponential phase
            if len(valid_y) >= 4:
                # Find the steepest growth period (highest slope)
                growth_rates = []
                for i in range(1, min(len(valid_y)-1, 5)):  # Check first few points
                    if valid_time[i+1] != valid_time[i-1]:
                        slope = (valid_y[i+1] - valid_y[i-1]) / (valid_time[i+1] - valid_time[i-1])
                        if valid_y[i] > 0:
                            specific_growth_rate = slope / valid_y[i]
                            growth_rates.append(specific_growth_rate)
                
                if growth_rates:
                    # Use median to avoid outliers
                    mu_est = np.median([rate for rate in growth_rates if rate > -10 and rate < 10])
                else:
                    mu_est = 0.1
            else:
                mu_est = 0.1
            
            # Estimate carrying capacity from final values
            y_final = np.mean(valid_y[-3:]) if len(valid_y) >= 3 else valid_y[-1]
            
            # Conservative self-interaction estimate
            if y_final > x0_est * 1.2:  # If there's growth limitation
                # Use a more conservative estimate
                a11_est = -mu_est / (y_final * 2)  # More conservative than just y_final
            else:
                a11_est = -0.001  # Very small negative value
            
            return x0_est, mu_est, a11_est
        
        # Only run fitting when button is clicked
        if st.session_state.get('run_fitting', False):
            st.session_state.run_fitting = False  # Reset flag
            
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
                    
                    # CRITICAL: Skip NA values in fitting - preserve data integrity
                    # Find valid (non-NA) data points
                    valid_mask = ~np.isnan(y_data)
                    valid_time = time_data[valid_mask]
                    valid_y = y_data[valid_mask]
                    
                    # Check if we have enough valid data points
                    if len(valid_y) < 3:
                        st.error(f"❌ {species}: Only {len(valid_y)} valid data points available. Need at least 3 for fitting.")
                        fit_results.append({
                            "Species": species, 
                            "Initial Value (x₀)": "Insufficient data", 
                            "Growth Rate (μ)": "Insufficient data", 
                            "Self-Interaction (a₁₁)": "Insufficient data",
                            "R²": "N/A",
                            "x0_val": np.nan,
                            "mu_val": np.nan,
                            "a11_val": np.nan,
                            "x0_CI_low": np.nan,
                            "x0_CI_high": np.nan,
                            "mu_CI_low": np.nan,
                            "mu_CI_high": np.nan,
                            "a11_CI_low": np.nan,
                            "a11_CI_high": np.nan
                        })
                        continue
                    
                    # Inform user about data usage
                    total_points = len(y_data)
                    valid_points = len(valid_y)
                    if valid_points < total_points:
                        st.info(f"📊 {species}: Using {valid_points}/{total_points} valid data points (skipped {total_points-valid_points} NA values)")
                    
                    # Verify lengths match for valid data
                    if len(valid_time) != len(valid_y):
                        st.error(f"Length mismatch for {species}: valid_time={len(valid_time)}, valid_y={len(valid_y)}")
                        continue
                    
                    # IMPROVED: Better initial parameter estimation from valid data  
                    x0_initial, mu_initial, a11_initial = estimate_logistic_params_better(valid_time, valid_y)
                    
                    # Calculate y_min, y_max for all cases (needed for fallback attempts)
                    y_min, y_max = np.min(valid_y), np.max(valid_y)
                    
                    try:
                        # Additional debugging for curve_fit with valid data only
                        st.write(f"Debug before curve_fit: valid_time length = {len(valid_time)}, valid_y length = {len(valid_y)}")
                        st.write(f"Debug: valid_time min/max = {valid_time.min():.3f}/{valid_time.max():.3f}")
                        st.write(f"Debug: valid_y min/max = {valid_y.min():.3f}/{valid_y.max():.3f}")
                        
                        # Test the logistic_model function before curve_fit with valid data
                        test_result = logistic_model(valid_time, valid_y[0], 0.1, -0.01)
                        st.write(f"Debug: logistic_model test successful, result length = {len(test_result)}")
                        
                        # Configure bounds based on selected mode
                        bounds_mode = st.session_state.get('bounds_mode', 'No Bounds (Unbounded)')
                        # Defaults to avoid NameError when checking bounds later
                        mu_min = mu_max = a11_min = a11_max = None
                        
                        if bounds_mode == "No Bounds (Unbounded)":
                            # No bounds - allow any parameter values
                            bounds_list = [(-np.inf, np.inf)] * 3  # For all 3 parameters
                            st.info(f"🔓 Fitting {species} with no parameter bounds")
                            
                        elif bounds_mode == "Auto Bounds":
                            # Auto bounds with reasonable defaults
                            mu_min, mu_max = -2.0, 2.0
                            a11_min, a11_max = -1.0, 0.1
                            # Data-driven x0 bounds
                            x0_lower = max(y_min * 0.1, 1e-8)
                            x0_upper = min(y_max * 10, y_max * 3.0)
                            bounds_list = ([x0_lower, mu_min, a11_min], [x0_upper, mu_max, a11_max])
                            st.info(f"🤖 Auto bounds for {species}: x₀=[{x0_lower:.3f}, {x0_upper:.3f}], μ=[{mu_min}, {mu_max}], a₁₁=[{a11_min}, {a11_max}]")
                            
                        elif bounds_mode == "Use Parameter Bounds":
                            # User-configured bounds
                            if "fitting_bounds" in st.session_state:
                                bounds_config = st.session_state.fitting_bounds
                                mu_min = bounds_config["mu_min"]
                                mu_max = bounds_config["mu_max"]
                                a11_min = bounds_config["a11_min"]
                                a11_max = bounds_config["a11_max"]
                                x0_mult = bounds_config["x0_multiplier"]
                                
                                # Use explicit x0 bounds if available, otherwise fall back to data-driven
                                if "x0_min" in bounds_config and "x0_max" in bounds_config and bounds_config["x0_min"] is not None:
                                    x0_lower = float(bounds_config["x0_min"])
                                    x0_upper = float(bounds_config["x0_max"])
                                    st.info(f"🔧 Using configured x₀ bounds for {species}: [{x0_lower:.6f}, {x0_upper:.6f}]")
                                else:
                                    # Fallback to data-driven bounds
                                    x0_lower = max(y_min * x0_mult, 1e-8)
                                    x0_upper = y_max / x0_mult
                                    st.info(f"📊 Using data-driven x₀ bounds for {species}: [{x0_lower:.6f}, {x0_upper:.6f}]")
                                
                                bounds_list = ([x0_lower, mu_min, a11_min], [x0_upper, mu_max, a11_max])
                            else:
                                # Fallback to auto bounds if no config
                                mu_min, mu_max = -2.0, 2.0
                                a11_min, a11_max = -1.0, 0.1
                                x0_lower = max(y_min * 0.1, 1e-8)
                                x0_upper = y_max * 3.0
                                bounds_list = ([x0_lower, mu_min, a11_min], [x0_upper, mu_max, a11_max])
                                st.warning(f"⚠️ No custom bounds found for {species}, using auto bounds")
                        
                        # Multiple optimization attempts with improved bounds
                        best_params = None
                        best_r2 = -np.inf
                        fit_success = False
                        
                        # Configure attempts based on bounds mode
                        if bounds_mode == "No Bounds (Unbounded)":
                            attempts = [
                                # Attempt 1: Conservative initial guess, no bounds
                                {
                                    'p0': [x0_initial, 0.1, -0.01],
                                    'bounds': bounds_list,  # No bounds
                                    'maxfev': 3000
                            },
                            # Attempt 2: Different initial guess
                            {
                                'p0': [valid_y[0], 0.01, -0.001],
                                'bounds': bounds_list,  # No bounds
                                'maxfev': 5000
                            },
                            # Attempt 3: Data-driven initial guess
                            {
                                'p0': [y_min, (y_max-y_min)/max(valid_time), -0.1],
                                'bounds': bounds_list,  # No bounds
                                'maxfev': 2000
                            }
                        ]
                        else:
                            # Bounded attempts
                            attempts = [
                                # Attempt 1: Conservative initial guess
                                {
                                    'p0': [x0_initial, 0.1, -0.01],
                                    'bounds': bounds_list,
                                    'maxfev': 3000,
                                    'method': 'trf'
                                },
                                # Attempt 2: Even more conservative
                                {
                                    'p0': [valid_y[0], 0.01, -0.001],
                                    'bounds': bounds_list,
                                    'maxfev': 5000,
                                    'method': 'trf'
                                },
                                # Attempt 3: Original approach as fallback
                                {
                                    'p0': [x0_initial, mu_initial, a11_initial],
                                    'bounds': bounds_list,
                                    'maxfev': 3000,
                                    'method': 'trf'
                                }
                            ]
                        
                        for attempt_num, params in enumerate(attempts):
                            try:
                                # Fit to valid data only
                                if bounds_mode == "No Bounds (Unbounded)":
                                    # No bounds fitting - don't use bounds parameter
                                    popt, pcov = curve_fit(logistic_model, valid_time, valid_y, 
                                                         p0=params['p0'], 
                                                         maxfev=params['maxfev'])
                                else:
                                    # Bounded fitting
                                    popt, pcov = curve_fit(logistic_model, valid_time, valid_y, 
                                                         p0=params['p0'], 
                                                         bounds=params['bounds'],
                                                         maxfev=params['maxfev'],
                                                         method=params.get('method', 'trf'))
                                
                                # Calculate R² for this attempt
                                y_pred = logistic_model(valid_time, *popt)
                                ss_res = np.sum((valid_y - y_pred) ** 2)
                                ss_tot = np.sum((valid_y - np.mean(valid_y)) ** 2)
                                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
                                
                                # Validate the fit
                                if np.all(np.isfinite(y_pred)) and np.all(y_pred > 0) and r2 > best_r2:
                                    best_params = popt
                                    best_r2 = r2
                                    best_pcov = pcov
                                    fit_success = True
                                    st.write(f"Debug: {species} fit successful on attempt {attempt_num + 1}, R²={r2:.4f}")
                                    
                            except Exception as fit_error:
                                st.write(f"Debug: {species} attempt {attempt_num + 1} failed: {fit_error}")
                                continue
                        
                        if fit_success and best_params is not None:
                            x0_fit, mu_fit, a11_fit = best_params
                            
                            # Enhanced parameter validation and boundary checking
                            bounds_hit = []
                            tolerance = 1e-6
                            
                            # Check if parameters hit bounds (indicates poor constraint)
                            if mu_min is not None and mu_max is not None:
                                if abs(mu_fit - mu_min) < tolerance:
                                    bounds_hit.append(f"μ hit lower bound ({mu_min})")
                                elif abs(mu_fit - mu_max) < tolerance:
                                    bounds_hit.append(f"μ hit upper bound ({mu_max})")
                            
                            if a11_min is not None and a11_max is not None:
                                if abs(a11_fit - a11_min) < tolerance:
                                    bounds_hit.append(f"a₁₁ hit lower bound ({a11_min})")
                                elif abs(a11_fit - a11_max) < tolerance:
                                    bounds_hit.append(f"a₁₁ hit upper bound ({a11_max})")
                            
                            # Calculate parameter errors with boundary checking
                            try:
                                perr = np.sqrt(np.diag(best_pcov))
                                x0_err, mu_err, a11_err = perr
                                
                                # Check for extremely large errors (indicates poor fit/boundary issues)
                                if mu_err > 1000 or a11_err > 1000:
                                    st.error(f"🚨 {species}: Extremely large parameter uncertainties detected!")
                                    st.write(f"   • μ error: {mu_err:.4f} (fitted: {mu_fit:.6f})")
                                    st.write(f"   • a₁₁ error: {a11_err:.4f} (fitted: {a11_fit:.6f})")
                                    if bounds_hit:
                                        st.write(f"   • Boundary issues: {', '.join(bounds_hit)}")
                                    st.write("   • **Suggestion**: Try wider bounds or check if data supports this model")
                                    
                            except Exception as err_calc_error:
                                st.warning(f"⚠️ {species}: Could not calculate parameter errors: {err_calc_error}")
                                x0_err = mu_err = a11_err = np.nan
                            
                            # Enhanced sanity check
                            warnings = []
                            if abs(mu_fit) > 5.0:
                                warnings.append(f"μ={mu_fit:.4f} seems extreme")
                            if abs(a11_fit) > 1.0:
                                warnings.append(f"a₁₁={a11_fit:.4f} seems extreme")
                            if bounds_hit:
                                warnings.extend(bounds_hit)
                                
                            if warnings:
                                st.warning(f"⚠️ {species}: " + "; ".join(warnings))
                                if bounds_hit:
                                    st.info("💡 **Tip**: If parameters hit bounds, try adjusting bounds or check data quality")
                            
                            r_squared = best_r2

                            # Confidence Intervals (95%)
                            x0_CI_low = x0_fit - 1.96 * x0_err if not np.isnan(x0_err) else np.nan
                            x0_CI_high = x0_fit + 1.96 * x0_err if not np.isnan(x0_err) else np.nan
                            mu_CI_low = mu_fit - 1.96 * mu_err if not np.isnan(mu_err) else np.nan
                            mu_CI_high = mu_fit + 1.96 * mu_err if not np.isnan(mu_err) else np.nan
                            a11_CI_low = a11_fit - 1.96 * a11_err if not np.isnan(a11_err) else np.nan
                            a11_CI_high = a11_fit + 1.96 * a11_err if not np.isnan(a11_err) else np.nan

                            fit_results.append({
                                "Species": species, 
                                "Initial Value (x₀)": f"{x0_fit:.4f}" + (f" ± {x0_err:.4f}" if not np.isnan(x0_err) else ""),
                                "Growth Rate (μ)": f"{mu_fit:.4f}" + (f" ± {mu_err:.4f}" if not np.isnan(mu_err) else ""),
                                "Self-Interaction (a₁₁)": f"{a11_fit:.4f}" + (f" ± {a11_err:.4f}" if not np.isnan(a11_err) else ""),
                                "R²": f"{r_squared:.4f}",
                                "x0_val": x0_fit,
                                "mu_val": mu_fit,
                                "a11_val": a11_fit,
                                "a11_se": a11_err,
                                "x0_CI_low": x0_CI_low,
                                "x0_CI_high": x0_CI_high,
                                "mu_CI_low": mu_CI_low,
                                "mu_CI_high": mu_CI_high,
                                "a11_CI_low": a11_CI_low,
                                "a11_CI_high": a11_CI_high
                            })
                        else:
                            # Fitting failed completely
                            st.error(f"❌ All fitting attempts failed for {species}")
                            fit_results.append({
                                "Species": species, 
                                "Initial Value (x₀)": "Failed", 
                                "Growth Rate (μ)": "Failed", 
                                "Self-Interaction (a₁₁)": "Failed",
                                "R²": "N/A",
                                "x0_val": np.nan,
                                "mu_val": np.nan,
                                "a11_val": np.nan,
                                "a11_se": np.nan,
                                "x0_CI_low": np.nan,
                                "x0_CI_high": np.nan,
                                "mu_CI_low": np.nan,
                                "mu_CI_high": np.nan,
                                "a11_CI_low": np.nan,
                                "a11_CI_high": np.nan
                            })
                    
                    except Exception as e:
                        fit_results.append({
                            "Species": species, 
                            "Initial Value (x₀)": "Failed", 
                            "Growth Rate (μ)": "Failed", 
                            "Self-Interaction (a₁₁)": "Failed",
                            "R²": "N/A",
                            "x0_val": np.nan,
                            "mu_val": np.nan,
                            "a11_val": np.nan,
                            "a11_se": np.nan,
                            "x0_CI_low": np.nan,
                            "x0_CI_high": np.nan,
                            "mu_CI_low": np.nan,
                            "mu_CI_high": np.nan,
                            "a11_CI_low": np.nan,
                            "a11_CI_high": np.nan
                        })
                        st.error(f"Fit failed for {species}: {str(e)}")
                        st.write(f"Debug: time_data shape: {time_data.shape}")
                        st.write(f"Debug: y_data shape: {y_data.shape}")
                        st.write(f"Debug: fit_indices: {fit_indices[:5]}...{fit_indices[-5:] if len(fit_indices) > 5 else ''}")
            
            df_fit = pd.DataFrame(fit_results)
            st.session_state.df_fit = df_fit
            st.session_state.fitted_parameters = df_fit  # For biological pairwise fitting        

        # Model Fitting Results section (always visible)
        st.subheader("Model Fitting Results")
        fit_tabs = st.tabs(["Parameter Table", "Fitted Curves", "Growth Rate Comparison", "LV Pairwise Theory", "LV Biological Fit", "LV Analysis & Plots"])
        with fit_tabs[0]:
            # Bounds Configuration Dropdown
            st.markdown("### 🎯 Fitting Configuration")
            
            # Initialize bounds mode in session state
            if "bounds_mode" not in st.session_state:
                st.session_state.bounds_mode = "No Bounds (Unbounded)"
            
            bounds_mode = st.selectbox(
                "Select bounds configuration:",
                ["No Bounds (Unbounded)", "Auto Bounds", "Use Parameter Bounds"],
                index=["No Bounds (Unbounded)", "Auto Bounds", "Use Parameter Bounds"].index(st.session_state.bounds_mode),
                help="Choose how to constrain parameter optimization"
            )
            st.session_state.bounds_mode = bounds_mode
            
            # Show explanation for each mode
            if bounds_mode == "No Bounds (Unbounded)":
                st.info("🔓 **No Bounds**: Parameters can take any values. May find unrealistic solutions but maximum flexibility.")
            elif bounds_mode == "Auto Bounds":
                st.info("🤖 **Auto Bounds**: System automatically sets reasonable parameter ranges based on your data. Good balance of robustness and simplicity.")
            elif bounds_mode == "Use Parameter Bounds":
                st.info("⚙️ **Custom Bounds**: You manually configure parameter ranges. Maximum control but requires domain knowledge.")
            
            # Initialize bounds in session state if not exists
            if "fitting_bounds" not in st.session_state:
                st.session_state.fitting_bounds = {
                    "mu_min": -2.0, "mu_max": 2.0,
                    "a11_min": -1.0, "a11_max": 0.1,
                    "x0_min": None, "x0_max": None,  # Will be set to data-dependent values
                    "x0_multiplier": 0.1  # x0 bounds will be data-dependent
                }
            
            # Show bounds configuration based on selected mode
            if bounds_mode == "Use Parameter Bounds":
                with st.expander("⚙️ Parameter Bounds Configuration", expanded=True):
                    st.markdown("**Set parameter bounds for more robust fitting:**")
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.markdown("**Growth Rate (μ) Bounds:**")
                        mu_min = st.number_input("μ minimum", 
                                                value=st.session_state.fitting_bounds["mu_min"], 
                                                step=0.01,  # Smaller step for precision
                                                format="%.8f",  # Show more decimal places
                                                help="Minimum allowed growth rate")
                        mu_max = st.number_input("μ maximum", 
                                                value=st.session_state.fitting_bounds["mu_max"], 
                                                step=0.01,
                                                format="%.8f",
                                                help="Maximum allowed growth rate")
                        
                    with col_b2:
                        st.markdown("**Self-Interaction (a₁₁) Bounds:**")
                        a11_min = st.number_input("a₁₁ minimum", 
                                                 value=st.session_state.fitting_bounds["a11_min"], 
                                                 step=0.001,  # Even smaller step for precision
                                                 format="%.8f",
                                                 help="Minimum allowed self-interaction (usually negative)")
                        a11_max = st.number_input("a₁₁ maximum", 
                                                 value=st.session_state.fitting_bounds["a11_max"], 
                                                 step=0.001,
                                                 format="%.8f", 
                                                 help="Maximum allowed self-interaction")
                    
                    with col_b3:
                        st.markdown("**Initial Value (x₀) Bounds:**")
                        
                        # Calculate data-driven defaults for x0 bounds
                        if 'data_proc' in st.session_state and st.session_state.data_proc is not None:
                            data_no_time = st.session_state.data_proc.drop(columns=['time'], errors='ignore')
                            if not data_no_time.empty:
                                data_min = data_no_time.min().min()
                                data_max = data_no_time.max().max()
                                first_values = data_no_time.iloc[0]
                                
                                # Default bounds: 10% to 10x of actual first values
                                default_x0_min = max(first_values.min() * 0.1, data_min * 0.05)
                                default_x0_max = min(first_values.max() * 10, data_max * 2)
                            else:
                                default_x0_min = 0.01
                                default_x0_max = 100.0
                        else:
                            default_x0_min = 0.01
                            default_x0_max = 100.0
                        
                        # Use stored values if available, otherwise defaults
                        current_x0_min = st.session_state.fitting_bounds.get("x0_min", default_x0_min)
                        current_x0_max = st.session_state.fitting_bounds.get("x0_max", default_x0_max)
                        
                        # Handle None values - use defaults if stored values are None
                        if current_x0_min is None:
                            current_x0_min = default_x0_min
                        if current_x0_max is None:
                            current_x0_max = default_x0_max
                        
                        x0_min = st.number_input("x₀ minimum", 
                                               value=float(current_x0_min), 
                                               step=0.01,
                                               min_value=0.001,
                                               format="%.8f",
                                               help="Minimum allowed initial value")
                        x0_max = st.number_input("x₀ maximum", 
                                               value=float(current_x0_max), 
                                               step=0.1,
                                               min_value=0.01,
                                               format="%.8f",
                                               help="Maximum allowed initial value")
                    
                    # Add scientific notation inputs for very small values
                    st.markdown("**🔬 Scientific Notation Input (for very small bounds):**")
                    st.markdown("*Use these if you need values smaller than 0.001*")
                    
                    col_sci1, col_sci2, col_sci3 = st.columns(3)
                    with col_sci1:
                        use_sci_mu = st.checkbox("Use scientific notation for μ bounds", 
                                                help="Enable for very small growth rate bounds")
                        if use_sci_mu:
                            mu_min_sci = st.text_input("μ minimum (scientific)", 
                                                      value=f"{mu_min:.2e}",
                                                      help="Example: -1e-6, 1.5e-8")
                            mu_max_sci = st.text_input("μ maximum (scientific)", 
                                                      value=f"{mu_max:.2e}",
                                                      help="Example: 1e-6, 2.5e-4")
                            try:
                                mu_min = float(mu_min_sci)
                                mu_max = float(mu_max_sci)
                                st.success(f"✅ μ bounds: [{mu_min:.2e}, {mu_max:.2e}]")
                            except ValueError:
                                st.error("❌ Invalid scientific notation. Use format like: 1e-6")
                    
                    with col_sci2:
                        use_sci_a11 = st.checkbox("Use scientific notation for a₁₁ bounds",
                                                 help="Enable for very small interaction bounds")
                        if use_sci_a11:
                            a11_min_sci = st.text_input("a₁₁ minimum (scientific)", 
                                                       value=f"{a11_min:.2e}",
                                                       help="Example: -1e-6, -5e-8")
                            a11_max_sci = st.text_input("a₁₁ maximum (scientific)", 
                                                       value=f"{a11_max:.2e}",
                                                       help="Example: 1e-6, 1e-4")
                            try:
                                a11_min = float(a11_min_sci)
                                a11_max = float(a11_max_sci)
                                st.success(f"✅ a₁₁ bounds: [{a11_min:.2e}, {a11_max:.2e}]")
                            except ValueError:
                                st.error("❌ Invalid scientific notation. Use format like: -1e-6")
                    
                    with col_sci3:
                        use_sci_x0 = st.checkbox("Use scientific notation for x₀ bounds",
                                               help="Enable for very small initial value bounds")
                        if use_sci_x0:
                            x0_min_sci = st.text_input("x₀ minimum (scientific)", 
                                                     value=f"{x0_min:.2e}",
                                                     help="Example: 1e-6, 5e-8")
                            x0_max_sci = st.text_input("x₀ maximum (scientific)", 
                                                     value=f"{x0_max:.2e}",
                                                     help="Example: 1e2, 5e3")
                            try:
                                x0_min = float(x0_min_sci)
                                x0_max = float(x0_max_sci)
                                st.success(f"✅ x₀ bounds: [{x0_min:.2e}, {x0_max:.2e}]")
                            except ValueError:
                                st.error("❌ Invalid scientific notation. Use format like: 1e-6")
                    
                    x0_mult = st.number_input("Initial value bounds multiplier", 
                                            value=st.session_state.fitting_bounds["x0_multiplier"], 
                                            step=0.01, 
                                            min_value=0.001,
                                            format="%.8f",
                                            help="x₀ bounds will be [data_min × multiplier, data_max / multiplier] (fallback method)")
                    
                    # Enhanced validation
                    if mu_min >= mu_max:
                        st.error("❌ μ minimum must be less than μ maximum")
                    elif a11_min >= a11_max:
                        st.error("❌ a₁₁ minimum must be less than a₁₁ maximum")
                    elif x0_min >= x0_max:
                        st.error("❌ x₀ minimum must be less than x₀ maximum")
                    else:
                        st.success(f"✅ Bounds valid: μ ∈ [{mu_min:.6f}, {mu_max:.6f}], a₁₁ ∈ [{a11_min:.6f}, {a11_max:.6f}], x₀ ∈ [{x0_min:.6f}, {x0_max:.6f}]")
                        
                        # Additional warnings for potentially problematic bounds
                        warnings = []
                        
                        # Check for very narrow bounds that might cause boundary issues
                        mu_range = mu_max - mu_min
                        a11_range = a11_max - a11_min
                        x0_range = x0_max - x0_min
                        
                        if mu_range < 0.01:
                            warnings.append(f"Very narrow μ range ({mu_range:.6f}) - may cause boundary fitting")
                        if a11_range < 0.001:
                            warnings.append(f"Very narrow a₁₁ range ({a11_range:.6f}) - may cause boundary fitting")
                        if x0_range < x0_min * 0.1:
                            warnings.append(f"Very narrow x₀ range ({x0_range:.6f}) - may cause boundary fitting")
                        
                        # Check for potentially unrealistic bounds
                        if a11_max <= 0 and a11_min < -0.5:
                            warnings.append("a₁₁ bounds only allow strong negative self-interaction - consider allowing small positive values")
                        if mu_max <= 0 and mu_min < -0.1:
                            warnings.append("μ bounds only allow negative growth - unusual for biological systems")
                        if x0_min <= 0:
                            warnings.append("x₀ minimum is zero or negative - initial populations should be positive")
                        
                        # Check if x0 bounds are reasonable for the data
                        if 'data_proc' in st.session_state and st.session_state.data_proc is not None:
                            data_no_time = st.session_state.data_proc.drop(columns=['time'], errors='ignore')
                            if not data_no_time.empty:
                                first_values = data_no_time.iloc[0]
                                if x0_max < first_values.max():
                                    warnings.append(f"x₀ maximum ({x0_max:.3f}) is smaller than largest actual initial value ({first_values.max():.3f})")
                                if x0_min > first_values.min():
                                    warnings.append(f"x₀ minimum ({x0_min:.3f}) is larger than smallest actual initial value ({first_values.min():.3f})")
                        
                        if warnings:
                            for warning in warnings:
                                st.warning(f"⚠️ {warning}")
                            st.info("💡 **Tip**: Very narrow bounds can force parameters to boundary values with high uncertainty")
                    
                    if st.button("🔄 Update Bounds & Refit"):
                        # Store the bounds with full precision
                        st.session_state.fitting_bounds = {
                            "mu_min": float(mu_min), 
                            "mu_max": float(mu_max),
                            "a11_min": float(a11_min), 
                            "a11_max": float(a11_max),
                            "x0_min": float(x0_min),
                            "x0_max": float(x0_max),
                            "x0_multiplier": float(x0_mult)
                        }
                        st.success("Bounds updated with full precision! Refitting models...")
                        st.info(f"🔍 **Applied bounds**: μ=[{mu_min:.2e}, {mu_max:.2e}], a₁₁=[{a11_min:.2e}, {a11_max:.2e}], x₀=[{x0_min:.2e}, {x0_max:.2e}]")
                        st.rerun()
            
            elif bounds_mode == "Auto Bounds":
                with st.expander("🤖 Auto Bounds Display", expanded=True):
                    st.markdown("**Automatically calculated parameter bounds:**")
                    
                    # Calculate auto bounds
                    auto_mu_min, auto_mu_max = -2.0, 2.0
                    auto_a11_min, auto_a11_max = -1.0, 0.1
                    
                    # Data-driven x0 bounds
                    if 'data_proc' in st.session_state and st.session_state.data_proc is not None:
                        data_no_time = st.session_state.data_proc.drop(columns=['time'], errors='ignore')
                        if not data_no_time.empty:
                            first_values = data_no_time.iloc[0]
                            auto_x0_min = max(first_values.min() * 0.1, data_no_time.min().min() * 0.05)
                            auto_x0_max = min(first_values.max() * 10, data_no_time.max().max() * 2)
                        else:
                            auto_x0_min, auto_x0_max = 0.01, 100.0
                    else:
                        auto_x0_min, auto_x0_max = 0.01, 100.0
                    
                    col_auto1, col_auto2, col_auto3 = st.columns(3)
                    with col_auto1:
                        st.metric("Growth Rate (μ)", f"[{auto_mu_min}, {auto_mu_max}]", "Fixed range")
                    with col_auto2:
                        st.metric("Self-Interaction (a₁₁)", f"[{auto_a11_min}, {auto_a11_max}]", "Competition focused") 
                    with col_auto3:
                        st.metric("Initial Value (x₀)", f"[{auto_x0_min:.3f}, {auto_x0_max:.3f}]", "Data adaptive")
                    
                    st.info("🤖 These bounds are automatically calculated based on biological reasonableness and your data range.")
            
            # Manual fitting buttons
            st.markdown("---")
            st.markdown("### 🚀 Run Fitting")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🎯 **Fit Individual Species**", type="primary"):
                    st.session_state.run_fitting = True
                    st.rerun()
            
            with col_btn2:
                if bounds_mode != "No Bounds (Unbounded)" and st.button("🔄 Update & Refit"):
                    st.session_state.run_fitting = True
                    st.rerun()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Modified Growth Model Parameters with 95% Confidence Intervals")
                st.markdown("**Model:** dX/dt = μ(X + a₁₁X) (Modified Growth)")
            with col2:
                # Check if we have fitted data
                if 'df_fit' in st.session_state and st.session_state.df_fit is not None:
                    df_fit = st.session_state.df_fit
                    st.markdown(download_csv(
                        df_fit[["Species", "Initial Value (x₀)", "Growth Rate (μ)", "Self-Interaction (a₁₁)", "R²", "x0_CI_low", "x0_CI_high", "mu_CI_low", "mu_CI_high", "a11_CI_low", "a11_CI_high"]], 
                        "fit_parameters_with_CI"), unsafe_allow_html=True)
                else:
                    st.info("Click 'Fit Individual Species' to generate results")
            
            # Display results if available
            if 'df_fit' in st.session_state and st.session_state.df_fit is not None:
                df_fit = st.session_state.df_fit
                st.dataframe(df_fit[[
                    "Species", 
                    "Initial Value (x₀)", "x0_CI_low", "x0_CI_high",
                    "Growth Rate (μ)", "mu_CI_low", "mu_CI_high",
                    "Self-Interaction (a₁₁)", "a11_CI_low", "a11_CI_high",
                    "R²"
                ]], use_container_width=True)
            else:
                st.info("💡 **No fitting results yet.** Click 'Fit Individual Species' to generate parameter estimates.")

        with fit_tabs[1]:
            # Plot controls
            show_ci = st.checkbox("Show Confidence Intervals", value=True, key="show_ci_individual")
            
            # Check if fitting results are available
            if 'df_fit' in st.session_state and st.session_state.df_fit is not None:
                df_fit = st.session_state.df_fit
                
                fig_fit = go.Figure()
                colors = px.colors.qualitative.Bold
                for idx, species in enumerate(species_cols):
                    # Get parameters from fits using species name lookup instead of index
                    fit_row = df_fit[df_fit["Species"] == species]
                    if fit_row.empty:
                        continue
                    
                    x0_fit = fit_row["x0_val"].values[0]
                    mu_fit = fit_row["mu_val"].values[0]
                    a11_fit = fit_row["a11_val"].values[0]
                    mu_CI_low = fit_row["mu_CI_low"].values[0]
                    mu_CI_high = fit_row["mu_CI_high"].values[0]

                    # Get the data with proper NA handling
                    y_full = data_no_time[species].values
                    valid_mask = ~np.isnan(y_full)
                    valid_time_full = time_data_full[valid_mask]
                    valid_y_full = y_full[valid_mask]

                    # Data curve - plot only valid points, gaps will show as breaks in line
                    fig_fit.add_trace(go.Scatter(
                        x=valid_time_full,  # Only valid time points
                        y=valid_y_full,     # Only valid y values
                        mode='lines+markers',
                        name=f"{species} data",
                        line=dict(color=colors[idx % len(colors)]), 
                        opacity=0.7,
                        connectgaps=False  # Important: don't connect across NA gaps!
                    ))

                    # Fit curve with CI bands for individual species logistic fits
                    if not np.isnan(x0_fit) and not np.isnan(mu_fit) and not np.isnan(a11_fit):
                        fitted_curve = logistic_model(time_data, x0_fit, mu_fit, a11_fit)
                        color_hex = colors[idx % len(colors)]
                        fig_fit.add_trace(go.Scatter(
                            x=time_data,
                            y=fitted_curve,
                            mode='lines',
                            name=f"{species} fit",
                            line=dict(color=color_hex, dash='dash', width=2)
                        ))
                        
                        # Add confidence intervals if available and enabled
                        if show_ci and not np.isnan(mu_CI_low) and not np.isnan(mu_CI_high):
                            # Upper confidence bound
                            fitted_curve_upper = logistic_model(time_data, x0_fit, mu_CI_high, a11_fit)
                            # Lower confidence bound  
                            fitted_curve_lower = logistic_model(time_data, x0_fit, mu_CI_low, a11_fit)
                            
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
                    title="Individual Species Data with Logistic Model Fit (NA values preserved as gaps)",
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
                st.success("✅ **Scientific Data Handling**: NA values preserved as gaps - fitted curves based on valid data points only!")
                st.markdown(get_svg_download_link(fig_fit, "fitted_curves_with_CI"), unsafe_allow_html=True)
            
            else:
                st.info("💡 **No fitting results available.** Click 'Fit Individual Species' in the Parameter Table tab to generate fitted curves.")

        with fit_tabs[2]:
            # Use results from session state if available
            if 'df_fit' in st.session_state and st.session_state.df_fit is not None:
                df_fit_current = st.session_state.df_fit
                growth_rates = df_fit_current["mu_val"].tolist()
                species_names = df_fit_current["Species"].tolist()
                colors = px.colors.qualitative.Bold
                
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
            else:
                st.info("💡 **No fitting results available.** Click 'Fit Individual Species' in the Parameter Table tab to generate growth rate comparisons.")

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
            st.subheader("🧮 LV Biological Pairwise Fitting")
            
            # Explanation of biological methodology
            st.info("""
            **🧬 Biological Pairwise Methodology:**
            This approach implements the correct biological fitting methodology where each pairwise combination 
            is fitted as an **independent 2-species system** rather than extracting from a full 5-species simulation.
            
            **Key Features:**
            - ✅ **Individual fitting**: Each X₁+X₂, X₁+X₃, etc. combination is fitted separately as a 2-species Lotka-Volterra system
            - ✅ **Soft priors on monoculture parameters**: μ_i, α_ii fitted with soft constraints to stay close to individual fits
            - ✅ **Cross-interaction estimation**: α_ij, α_ji fitted for each pair with data-driven bounds
            - ✅ **Proper initial conditions**: Uses pairwise culture initial conditions, not full community
            - ✅ **Individual species data**: Fits to actual X₁(X₁+X₂) and X₂(X₁+X₂) counts when available
            
            This is biologically correct because pairwise experiments only contain 2 species, not all 5.
            """)

            # Debug: Check focal species configuration and data availability
            st.markdown("### 🔍 **Debug: Focal Species & Data Availability**")

            data_format = st.session_state.get('data_format', 'format_1_total_pairs')
            data_structure = st.session_state.get('data_structure', {})

            if data_format == 'format_2_focal_individual':
                # Show focal species configuration
                focal_species = st.session_state.get('focal_species', 'Unknown')
                st.info(f"**Configured Focal Species:** {focal_species}")
                
                # Show what columns are actually in your data
                pairwise_data = data_structure.get('pairwise_combinations', {})
                st.markdown("**Available Pairwise Data Columns:**")
                
                focal_pairs_found = []
                other_pairs_found = []
                
                for col_name, col_info in pairwise_data.items():
                    if col_info['type'] == 'individual_count_in_pair':
                        species_indices = col_info['species_indices']
                        individual_species = col_info['individual_species']
                        
                        # FIXED: Check if this involves the focal species (X1 = species index 1, not 0)
                        if 1 in species_indices:  # X1 is focal species (species index 1)
                            focal_pairs_found.append(f"✅ `{col_name}` - Species {individual_species} in pair {species_indices}")
                        else:
                            other_pairs_found.append(f"❌ `{col_name}` - Species {individual_species} in pair {species_indices} (NO X1)")
                
                st.markdown("**Focal Species (X1) Pairwise Data Found:**")
                if focal_pairs_found:
                    for pair in focal_pairs_found:
                        st.write(pair)
                    st.success(f"✅ **Found {len(focal_pairs_found)} X1 pairwise data columns!** The data is correctly available.")
                else:
                    st.error("🚨 **NO X1 pairwise data found!** This explains why you only see X2 combinations.")
                
                st.markdown("**Other Pairwise Data (Non-focal):**")
                if other_pairs_found:
                    for pair in other_pairs_found:
                        st.write(pair)
                
                # Show the actual column mapping that was applied
                if 'column_mapping' in st.session_state:
                    st.markdown("**Original Column Mapping:**")
                    mapping = st.session_state.column_mapping
                    focal_related_mappings = {k: v for k, v in mapping.items() if 'Species 1' in k}
                    if focal_related_mappings:
                        for data_type, file_column in focal_related_mappings.items():
                            st.write(f"• {data_type} → `{file_column}`")
                    else:
                        st.warning("No Species 1 related mappings found in column mapping!")

            else:
                st.info(f"Current format: {data_format} - Focal species analysis only applies to Format 2")

            # --- Individual Logistic Fits Overview (added first) ---
            st.markdown("### 🔍 Individual Species Logistic Fits (Reference)")
            if 'df_fit' in st.session_state and st.session_state.df_fit is not None:
                df_fit_ref = st.session_state.df_fit
                # Show compact table summary
                st.dataframe(df_fit_ref[["Species", "Initial Value (x₀)", "Growth Rate (μ)", "Self-Interaction (a₁₁)", "R²"]], use_container_width=True, height=200)

                # Plot all individual fits together
                try:
                    if 'df_avg' in st.session_state:
                        df_avg_ref = st.session_state.df_avg
                        time_vals = df_avg_ref['time'].values
                        species_cols_plot = st.session_state.species_cols
                        fig_indiv = go.Figure()
                        colors_local = px.colors.qualitative.Dark24 if len(species_cols_plot) > 10 else px.colors.qualitative.Bold
                        for idx, sp in enumerate(species_cols_plot):
                            # Observed
                            if sp in df_avg_ref.columns:
                                fig_indiv.add_trace(go.Scatter(x=time_vals, y=df_avg_ref[sp].values,
                                                               mode='markers', name=f"{sp} data",
                                                               marker=dict(color=colors_local[idx % len(colors_local)], size=6, symbol='circle-open')))
                            # Fitted curve
                            row = df_fit_ref[df_fit_ref['Species'] == sp]
                            if not row.empty and not np.isnan(row['mu_val'].values[0]) and not np.isnan(row['a11_val'].values[0]):
                                x0_fit = row['x0_val'].values[0]
                                mu_fit = row['mu_val'].values[0]
                                a11_fit = row['a11_val'].values[0]
                                fitted_curve = logistic_model(time_vals, x0_fit, mu_fit, a11_fit)
                                fig_indiv.add_trace(go.Scatter(x=time_vals, y=fitted_curve,
                                                               mode='lines', name=f"{sp} fit",
                                                               line=dict(color=colors_local[idx % len(colors_local)], width=2)))
                        fig_indiv.update_layout(title="Individual Species Logistic Fits",
                                                xaxis_title="Time",
                                                yaxis_title="Population",
                                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                                                template='plotly_white')
                        st.plotly_chart(fig_indiv, use_container_width=True)
                    else:
                        st.info("Raw averaged data (df_avg) not available for plotting.")
                except Exception as e:
                    st.warning(f"Could not render individual fit plot: {e}")
            else:
                st.info("Run individual species fitting first to see overview plot here.")
            
            # Data Format Analysis and Explanation
            data_format = st.session_state.get('data_format', 'format_1_total_pairs')
            data_structure = st.session_state.get('data_structure', {})
            
            with st.expander("� **Biological Pairwise Fitting Methodology**", expanded=True):
                st.markdown("""
                **This implements the proper biological approach for pairwise Lotka-Volterra fitting.**
                
                ### 🎯 **Method Overview:**
                
                **Step 1:** Use individual species parameters (μᵢ, aᵢᵢ) from Tab 1
                
                **Step 2:** For each pairwise culture (e.g., X₁+X₂):
                - Fit only 2 species: `dX/dt = μ(X + AX)` where X₃=X₄=X₅=0
                - Use data: X₁(X₁+X₂) and X₂(X₁+X₂) 
                - Estimate: α₁₂ and α₂₁ (cross-interactions)
                - Fix: μ₁, μ₂, α₁₁, α₂₂ from individual fits
                
                **Step 3:** Combine all pairwise results into full α matrix
                
                **Step 4:** Validate with full community data
                """)
                
                st.latex(r"""
                \text{For pairwise } X_1 + X_2: \quad
                \frac{d}{dt}\begin{bmatrix} X_1 \\ X_2 \\ 0 \\ 0 \\ 0 \end{bmatrix} = 
                \begin{bmatrix} \mu_1 \\ \mu_2 \\ 0 \\ 0 \\ 0 \end{bmatrix} \circ 
                \left(\begin{bmatrix} X_1 \\ X_2 \\ 0 \\ 0 \\ 0 \end{bmatrix} + 
                \begin{bmatrix} 
                \alpha_{11} & \alpha_{12} & 0 & 0 & 0 \\
                \alpha_{21} & \alpha_{22} & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0
                \end{bmatrix}
                \begin{bmatrix} X_1 \\ X_2 \\ 0 \\ 0 \\ 0 \end{bmatrix}\right)
                """)
            
            # Data format check
            with st.expander("�📊 **Data Format Analysis**", expanded=False):
                st.markdown(f"**Current Data Format:** `{data_format.replace('_', ' ').title()}`")
                
                if data_format == 'format_1_total_pairs':
                    st.error("""
                    **🔴 Format 1 - Cannot Use Biological Fitting:**
                    - Only has pairwise sums (X₁+X₂), not individual counts
                    - Need individual species data in pairwise cultures
                    - **Recommendation:** Upload Format 2 or 3 data for biological fitting
                    """)
                elif data_format == 'format_2_focal_species':
                    st.success("""
                    **� Format 2 - Perfect for Biological Fitting:**
                    - Has individual species counts: X₁(X₁+X₂), X₂(X₁+X₂), etc.
                    - ✅ Can implement proper pairwise fitting methodology
                    """)
                elif data_format == 'format_3_all_individual':
                    st.success("""
                    **🟢 Format 3 - Excellent for Biological Fitting:**
                    - Complete individual species data in all combinations
                    - ✅ Can implement full biological fitting methodology
                    """)
            
            # Check if individual species parameters are available for biological fitting
            if 'fitted_parameters' in st.session_state:
                df_fit = st.session_state.fitted_parameters
                st.markdown("### 🎯 **Individual Species Parameters Available:**")
                st.dataframe(df_fit[["Species", "Growth Rate (μ)", "Self-Interaction (a₁₁)", "R²"]], use_container_width=True)
                
                st.success("✅ **Ready for Biological Pairwise Fitting!**")
                st.markdown("""
                **New Biological Methodology Available:**
                - Individual species parameters loaded ✅
                - Format 2/3 data detected ✅  
                - Proper pairwise fitting can proceed ✅
                
                **Next Steps:** Implementation of biological pairwise fitting follows the exact methodology described.
                """)
            else:
                st.warning("⚠️ **Need Individual Species Parameters First**")
                st.markdown("""
                Please complete individual species fitting first:
                1. Go to **Tab 1: Parameter Table** 
                2. Run the fitting to get μᵢ and aᵢᵢ parameters
                3. Return here for biological pairwise fitting
                """)
            
            # Show available data summary
            if data_structure:
                pairwise_data = data_structure.get('pairwise_combinations', {})
                individual_data = data_structure.get('individual_species', {})
                
                st.markdown("### 📊 **Available Data Summary:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Individual Species", len(individual_data))
                with col2:
                    st.metric("Pairwise Combinations", len(pairwise_data))
            
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
                st.info("No monoculture fits found. We'll use editable priors for μ and a_ii.")

            # Prepare priors and initial conditions
            mu_priors = []
            aii_priors = []
            x0_vals = []

            # Smart initial condition estimation
            time_span = df_avg["time"].values
            x0_smart, x0_info = estimate_initial_conditions_smart(data_proc, species_cols, time_span)

            # Simple helper to estimate μ from co-culture components (first two positive points)
            def estimate_mu_prior_from_pairs(i_idx):
                try:
                    t = df_avg["time"].values
                    eps = 1e-9
                    slopes = []
                    for col_name, col_info in data_structure.get('pairwise_combinations', {}).items():
                        if (col_info.get('type') == 'individual_count_in_pair' and
                            (i_idx + 1) in tuple(col_info.get('species_indices', ()) or ())):
                            ind_sp_1based = col_info.get('individual_species')
                            if ind_sp_1based is None:
                                continue
                            if int(ind_sp_1based) - 1 != i_idx:
                                continue
                            series = np.asarray(data_proc[col_name].values, dtype=float)
                            series = np.clip(series, 0.0, None)
                            # find first two strictly positive entries
                            pos_idx = np.where(series > 0)[0]
                            if len(pos_idx) >= 2:
                                i1, i2 = pos_idx[0], pos_idx[1]
                                if np.isfinite(series[i1]) and np.isfinite(series[i2]) and t[i2] != t[i1]:
                                    mu_est = (np.log(series[i2] + eps) - np.log(series[i1] + eps)) / (t[i2] - t[i1])
                                    if np.isfinite(mu_est):
                                        slopes.append(mu_est)
                    if len(slopes) > 0:
                        # clamp to a reasonable range
                        return float(np.clip(np.median(slopes), -1.0, 1.0))
                except Exception:
                    pass
                return None

            # Display initial condition handling info
            with st.expander("🔍 Smart Initial Condition Handling", expanded=False):
                st.write("**How initial conditions were determined:**")
                for info in x0_info:
                    st.write(f"• {info}")

            # Load saved priors if available
            saved_priors = st.session_state.get("lv_priors")
            prior_sources = []
            for idx, sp in enumerate(species_cols):
                # μ prior
                mu_val = None
                src = "mono"
                if df_fit is not None:
                    mu_arr = df_fit.loc[df_fit["Species"] == sp, "mu_val"].values
                    if len(mu_arr) and np.isfinite(mu_arr[0]):
                        mu_val = float(mu_arr[0])
                if mu_val is None:
                    est = estimate_mu_prior_from_pairs(idx)
                    if est is not None:
                        mu_val, src = float(est), "co-culture estimate"
                if mu_val is None:
                    mu_val, src = 0.1, "default"
                if saved_priors and "mu" in saved_priors and sp in saved_priors["mu"]:
                    mu_val, src = float(saved_priors["mu"][sp]), "user"
                mu_priors.append(mu_val)
                prior_sources.append((sp, src))

                # a_ii prior
                aii_val = None
                src_aii = "mono"
                if df_fit is not None:
                    aii_arr = df_fit.loc[df_fit["Species"] == sp, "a11_val"].values
                    if len(aii_arr) and np.isfinite(aii_arr[0]):
                        aii_val = float(aii_arr[0])
                if aii_val is None:
                    aii_val, src_aii = -0.01, "default"
                if saved_priors and "aii" in saved_priors and sp in saved_priors["aii"]:
                    aii_val, src_aii = float(saved_priors["aii"][sp]), "user"
                aii_priors.append(aii_val)

                # Use smart initial value instead of simple iloc[0]
                initial_val = x0_smart[idx]
                if not np.isfinite(initial_val) or initial_val <= 0:
                    initial_val = 1e-6
                x0_vals.append(initial_val)

            mu_priors = np.array(mu_priors, dtype=float)
            aii_priors = np.array(aii_priors, dtype=float)
            x0_vals = np.array(x0_vals, dtype=float)

            # Priors editor UI
            if "lv_prior_lambda" not in st.session_state:
                st.session_state.lv_prior_lambda = 0.1

            with st.expander("🧭 Priors for μ and a_ii (used when monoculture fits are missing)", expanded=True):
                st.write("Edit priors for growth rates (μ) and self-interactions (a_ii). These anchor the LV fit via a soft penalty.")
                st.caption("Sources: " + ", ".join([f"{sp}:{src}" for sp, src in prior_sources]))
                with st.form("lv_priors_form"):
                    cols = st.columns(min(4, max(1, len(species_cols))))
                    mu_inputs = {}
                    aii_inputs = {}
                    for i, sp in enumerate(species_cols):
                        with cols[i % len(cols)]:
                            mu_inputs[sp] = st.number_input(f"μ prior – {sp}", value=float(mu_priors[i]), format="%.6f", key=f"mu_prior_{sp}")
                    adv = st.checkbox("Advanced: Edit a_ii priors", value=False, key="edit_aii_priors")
                    if adv:
                        cols2 = st.columns(min(4, max(1, len(species_cols))))
                        for i, sp in enumerate(species_cols):
                            with cols2[i % len(cols2)]:
                                aii_inputs[sp] = st.number_input(f"a_ii prior – {sp}", value=float(aii_priors[i]), format="%.6f", key=f"aii_prior_{sp}")
                    prior_lambda = st.slider("Prior strength λ (higher = stronger pull to priors)", min_value=0.0, max_value=2.0, value=float(st.session_state.lv_prior_lambda), step=0.05)
                    if st.form_submit_button("Apply Priors"):
                        # persist
                        new_mu = {sp: float(mu_inputs.get(sp, mu_priors[i])) for i, sp in enumerate(species_cols)}
                        new_aii = {sp: float(aii_inputs.get(sp, aii_priors[i])) for i, sp in enumerate(species_cols)}
                        st.session_state.lv_priors = {"mu": new_mu, "aii": new_aii}
                        st.session_state.lv_prior_lambda = float(prior_lambda)
                        st.success("Priors updated.")
                        # update local arrays for this run
                        mu_priors = np.array([new_mu[sp] for sp in species_cols], dtype=float)
                        aii_priors = np.array([new_aii[sp] for sp in species_cols], dtype=float)
                # reflect slider if not submitted
                prior_lambda = float(st.session_state.lv_prior_lambda)

            time_span = df_avg["time"].values
            n_species = len(species_cols)
            colors = px.colors.qualitative.Bold

            from itertools import combinations
            
            # Generate pair indices based on data format
            if data_format == 'format_2_focal_individual':
                # For Format 2, only include pairs involving the focal species
                focal_species = st.session_state.get('focal_species', 'Species 1')
                focal_index = int(focal_species.split()[-1]) - 1  # Convert "Species 1" to index 0
                
                # Only create pairs involving the focal species
                pair_indices = []
                for j in range(n_species):
                    if j != focal_index:
                        # Create pair (focal_index, j) in sorted order
                        if focal_index < j:
                            pair_indices.append((focal_index, j))
                        else:
                            pair_indices.append((j, focal_index))
                
                st.info(f"🎯 **Format 2 - Focal Species {focal_species}**: Fitting {len(pair_indices)} pairs involving focal species")
                for i, j in pair_indices:
                    st.write(f"• {species_cols[i]} + {species_cols[j]}")
            else:
                # For other formats, use all possible pairs
                pair_indices = list(combinations(range(n_species), 2))

            def lv_ode(t, x, mu, A):
                dxdt = np.zeros_like(x)
                for i in range(n_species):
                    dxdt[i] = x[i] * (mu[i] + A[i] @ x)
                return dxdt

            # a_ii priors are already prepared above; bounds during optimization enforce a_ii <= 0
            
            # --- Advanced optimization options (set defaults if not present) ---
            if 'lv_error_metric' not in st.session_state:
                st.session_state.lv_error_metric = 'normalized_mse'
            if 'lv_reg_lambda' not in st.session_state:
                st.session_state.lv_reg_lambda = 0.0
            if 'lv_use_log_errors' not in st.session_state:
                st.session_state.lv_use_log_errors = False
            if 'lv_pair_weighting' not in st.session_state:
                st.session_state.lv_pair_weighting = 'equal'

            # Helper to collect pair component series (Format 2/3) and measured initials in i,j order
            def get_pair_components_and_initials(i_idx, j_idx):
                try:
                    species_pair_indices_1based = tuple(sorted([i_idx + 1, j_idx + 1]))
                    obs_i = None
                    obs_j = None
                    for col_name, col_info in data_structure.get('pairwise_combinations', {}).items():
                        if (col_info.get('type') == 'individual_count_in_pair' and
                            tuple(sorted(col_info.get('species_indices', ()))) == species_pair_indices_1based):
                            ind_sp_1based = col_info.get('individual_species')
                            if ind_sp_1based is None:
                                continue
                            ind_sp_0based = int(ind_sp_1based) - 1
                            series = np.asarray(data_proc[col_name].values, dtype=float)
                            series = np.clip(series, 0.0, None)
                            if ind_sp_0based == i_idx:
                                obs_i = series
                            elif ind_sp_0based == j_idx:
                                obs_j = series
                    if obs_i is None or obs_j is None:
                        return None, None, None
                    # measured initials from first timepoint of each component
                    x0_pair = np.array([float(obs_i[0]), float(obs_j[0])], dtype=float)
                    return obs_i, obs_j, x0_pair
                except Exception:
                    return None, None, None

            def lv_residual_function(all_params):
                """
                Returns concatenated residuals from all available pairs with soft priors on monoculture parameters
                Parameter vector: [μ₀, a₀₀, μ₁, a₁₁, ..., α₀₁, α₁₀, α₀₂, α₂₀, ...]
                """
                all_residuals = []

                # Extract parameters from vector
                param_idx = 0

                # Extract monoculture parameters (μ_i, a_ii) for each species
                mu_fitted = []
                aii_fitted = []
                for i in range(n_species):
                    mu_i = all_params[param_idx]
                    aii_i = all_params[param_idx + 1]
                    mu_fitted.append(mu_i)
                    aii_fitted.append(aii_i)
                    param_idx += 2

                # Extract cross-interaction parameters (α_ij, α_ji) for each pair
                pairwise_params = {}
                for i, j in pair_indices:
                    alpha_ij = all_params[param_idx]
                    alpha_ji = all_params[param_idx + 1]
                    pairwise_params[(i, j)] = (alpha_ij, alpha_ji)
                    param_idx += 2

                # Add soft priors for monoculture parameters (encourage staying close to priors)
                prior_weight = float(st.session_state.get("lv_prior_lambda", 0.1))
                if prior_weight > 0:
                    for i in range(n_species):
                        # Prior penalty for μ_i (Gaussian prior centered on prior)
                        mu_prior_penalty = prior_weight * (mu_fitted[i] - mu_priors[i])**2
                        all_residuals.append(mu_prior_penalty)

                        # Prior penalty for a_ii (Gaussian prior centered on prior)
                        aii_prior_penalty = prior_weight * (aii_fitted[i] - aii_priors[i])**2
                        all_residuals.append(aii_prior_penalty)

                # Process each available pair
                for i, j in pair_indices:
                    try:
                        alpha_ij, alpha_ji = pairwise_params[(i, j)]
                        if not (np.isfinite(alpha_ij) and np.isfinite(alpha_ji)):
                            # Integration failure: return large residuals of correct length
                            all_residuals.extend([1e6] * (2 * len(time_span)))
                            continue

                        # Get component data
                        obs_i, obs_j, x0_pair = get_pair_components_and_initials(i, j)
                        if obs_i is None or obs_j is None or x0_pair is None:
                            # Skip if components missing - no residuals added for this pair
                            continue

                        # Build pair-specific 2x2 system with fitted parameters
                        aii_i = aii_fitted[i]  # Use fitted a_ii instead of fixed
                        aii_j = aii_fitted[j]  # Use fitted a_jj instead of fixed
                        A_pair = np.array([[aii_i, alpha_ij], [alpha_ji, aii_j]], dtype=float)
                        mu_pair = np.array([mu_fitted[i], mu_fitted[j]], dtype=float)  # Use fitted μ instead of fixed

                        # Safety on initials
                        x0_pair = np.clip(np.asarray(x0_pair, dtype=float), 1e-9, None)

                        # Define and solve ODE
                        def pairwise_lv_ode(t, x):
                            x1, x2 = x
                            dx1 = x1 * (mu_pair[0] + A_pair[0, 0] * x1 + A_pair[0, 1] * x2)
                            dx2 = x2 * (mu_pair[1] + A_pair[1, 0] * x1 + A_pair[1, 1] * x2)
                            return [dx1, dx2]

                        max_step = np.diff(time_span).max() if len(time_span) > 1 else None
                        sol_pair = solve_ivp(
                            pairwise_lv_ode,
                            (time_span[0], time_span[-1]),
                            x0_pair,
                            t_eval=time_span,
                            method="RK45",
                            rtol=1e-6,
                            atol=1e-9,
                            max_step=max_step
                        )

                        if not sol_pair.success:
                            # Integration failure: return large residuals of correct length
                            all_residuals.extend([1e6] * (2 * len(time_span)))
                            continue

                        sim_i, sim_j = sol_pair.y[0], sol_pair.y[1]
                        if not (np.all(np.isfinite(sim_i)) and np.all(np.isfinite(sim_j))):
                            # Numerical issues: return large residuals
                            all_residuals.extend([1e6] * (2 * len(time_span)))
                            continue

                        # Scaling: s_i = max(x_i_coc), s_j = max(x_j_coc)
                        s_i = float(np.max(obs_i))
                        s_j = float(np.max(obs_j))
                        s_i = s_i if s_i > 1e-9 else 1.0
                        s_j = s_j if s_j > 1e-9 else 1.0

                        # Scaled residuals: (x_i_hat - x_i)/s_i, (x_j_hat - x_j)/s_j
                        residuals_i = (sim_i - obs_i) / s_i
                        residuals_j = (sim_j - obs_j) / s_j

                        # Add to stacked residual vector
                        all_residuals.extend(residuals_i)
                        all_residuals.extend(residuals_j)

                    except Exception:
                        # Any other error: return large residuals
                        all_residuals.extend([1e6] * (2 * len(time_span)))
                        continue

                # Return as numpy array (required by least_squares)
                return np.array(all_residuals, dtype=float)

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
                    # Default: NO diagonal constraints since aii comes from individual fits
                    st.session_state.constraint_matrix_negative = [[False for j in range(n_species)] for i in range(n_species)]
                
                if "constraint_matrix_positive" not in st.session_state:
                    st.session_state.constraint_matrix_positive = [[False for j in range(n_species)] for i in range(n_species)]
                
                # Ensure matrices have correct dimensions (in case species count changed)
                if (len(st.session_state.constraint_matrix_negative) != n_species or 
                    len(st.session_state.constraint_matrix_negative[0]) != n_species):
                    st.session_state.constraint_matrix_negative = [[False for j in range(n_species)] for i in range(n_species)]
                
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
                                if i == j:  # Diagonal element
                                    st.write(f"α({species_cols[i]},{species_cols[j]})")
                                    st.info("Fitted with soft prior from individual fit")
                                    # Don't show checkboxes for diagonal - they're fitted with soft priors
                                    row_neg.append(False)
                                    row_pos.append(False)
                                else:  # Off-diagonal element
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

            # Initialize parameters for biological pairwise fitting
            # Fit ALL parameters: μ_i, a_ii for each species, plus α_ij, α_ji for each pair
            n_species_params = n_species * 2  # μ_i and a_ii for each species
            n_pairwise_params = len(pair_indices) * 2  # α_ij and α_ji for each pair
            n_total_params = n_species_params + n_pairwise_params

            # Initialize parameter vector and bounds
            init_params = []
            bounds_list = []

            # Add monoculture parameters (μ_i, a_ii) for each species - use priors as starting guesses
            for i in range(n_species):
                # Growth rate μ_i - use prior as starting guess
                mu_guess = float(mu_priors[i]) if np.isfinite(mu_priors[i]) else 0.1
                init_params.append(mu_guess)
                bounds_list.append((-2.0, 2.0))  # Reasonable μ bounds

                # Self-interaction a_ii - use prior as starting guess
                aii_guess = float(aii_priors[i]) if np.isfinite(aii_priors[i]) else -0.01
                init_params.append(aii_guess)
                bounds_list.append((-1.0, 0.0))  # a_ii must be negative

            # Add cross-interaction parameters (α_ij, α_ji) for each pair
            for i, j in pair_indices:
                # Get component data to calculate data-driven bounds
                obs_i, obs_j, x0_pair = get_pair_components_and_initials(i, j)
                if obs_i is not None and obs_j is not None:
                    # Data-driven bounds: ensure a_ij * max(x_j) stays in [-0.5, 0.5]
                    max_x_i = float(np.max(obs_i))
                    max_x_j = float(np.max(obs_j))

                    # Bounds for a_ij (effect of j on i)
                    bound_ij_low = -0.5 / max_x_j if max_x_j > 1e-9 else -1.0
                    bound_ij_high = 0.5 / max_x_j if max_x_j > 1e-9 else 1.0

                    # Bounds for a_ji (effect of i on j)
                    bound_ji_low = -0.5 / max_x_i if max_x_i > 1e-9 else -1.0
                    bound_ji_high = 0.5 / max_x_i if max_x_i > 1e-9 else 1.0
                else:
                    # Fallback bounds if no data
                    bound_ij_low, bound_ij_high = -1.0, 1.0
                    bound_ji_low, bound_ji_high = -1.0, 1.0

                # Apply user constraints if any
                is_negative_ij = st.session_state.constraint_matrix_negative[i][j]
                is_positive_ij = st.session_state.constraint_matrix_positive[i][j]
                is_negative_ji = st.session_state.constraint_matrix_negative[j][i]
                is_positive_ji = st.session_state.constraint_matrix_positive[j][i]

                # For a_ij
                if is_negative_ij:
                    bound_ij_high = min(bound_ij_high, 0.0)
                    init_params.append(-0.01)
                elif is_positive_ij:
                    bound_ij_low = max(bound_ij_low, 0.0)
                    init_params.append(0.01)
                else:
                    init_params.append(0.0)  # Start neutral

                bounds_list.append((bound_ij_low, bound_ij_high))

                # For a_ji
                if is_negative_ji:
                    bound_ji_high = min(bound_ji_high, 0.0)
                    init_params.append(-0.01)
                elif is_positive_ji:
                    bound_ji_low = max(bound_ji_low, 0.0)
                    init_params.append(0.01)
                else:
                    init_params.append(0.0)  # Start neutral

                bounds_list.append((bound_ji_low, bound_ji_high))

            init_params = np.array(init_params)
            
            # Show bounds information
            with st.expander("📊 Data-Driven Bounds Information", expanded=False):
                st.markdown("**Calculated bounds ensure that |a_ij × max(x_j)| ≤ 0.5 for biological realism:**")
                for idx, (i, j) in enumerate(pair_indices):
                    obs_i, obs_j, _ = get_pair_components_and_initials(i, j)
                    if obs_i is not None and obs_j is not None:
                        max_i, max_j = np.max(obs_i), np.max(obs_j)
                        st.write(f"• **{species_cols[i]}+{species_cols[j]}**: max({species_cols[i]})={max_i:.3f}, max({species_cols[j]})={max_j:.3f}")
                        alpha_start = 2 * n_species
                        st.write(f"  - a_{i+1}{j+1} ∈ [{bounds_list[alpha_start + idx*2][0]:.4f}, {bounds_list[alpha_start + idx*2][1]:.4f}]")
                        st.write(f"  - a_{j+1}{i+1} ∈ [{bounds_list[alpha_start + idx*2+1][0]:.4f}, {bounds_list[alpha_start + idx*2+1][1]:.4f}]")

            col1, col2 = st.columns([3, 1])
            with col1:
                run_lv = st.button("Run LV Biological Pairwise Fit")
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
                with st.spinner("Fitting α matrix using least_squares with soft_l1 loss..."):
                    if not st.session_state.lv_fit_done:
                        # Disallow totals-only format for biological component-based fitting
                        if data_format == 'format_1_total_pairs':
                            st.error("Biological pairwise fitting requires component tracks (Formats 2 or 3). Upload data with individual species in each pair.")
                            st.stop()
                        # Quick guard: ensure we have component data for at least one pair in formats 2/3
                        if data_format != 'format_1_total_pairs':
                            available_pairs = 0
                            for (ii_chk, jj_chk) in pair_indices:
                                oi, oj, x0p = get_pair_components_and_initials(ii_chk, jj_chk)
                                if oi is not None and oj is not None and x0p is not None:
                                    available_pairs += 1
                            if available_pairs == 0:
                                st.error("No pair component tracks found. Biological pairwise fitting requires component series in Format 2/3.")
                                st.stop()
                        
                        # Test the residual function first
                        try:
                            test_residuals = lv_residual_function(init_params)
                            if not np.all(np.isfinite(test_residuals)):
                                st.error(f"Residual function contains non-finite values. Check your data.")
                                st.stop()
                            test_cost = 0.5 * np.sum(test_residuals**2)
                            st.info(f"Initial cost function value: {test_cost:.6f} (from {len(test_residuals)} residuals)")
                        except Exception as e:
                            st.error(f"Residual function failed with initial values: {str(e)}")
                            st.stop()
                        
                        # ChatGPT's exact approach: use least_squares with soft_l1 loss
                        optimization_success = False
                        res = None
                        
                        try:
                            st.info("🚀 Running least_squares optimization with soft priors on monoculture parameters...")
                            res = least_squares(
                                lv_residual_function,
                                init_params,
                                bounds=(
                                    [b[0] for b in bounds_list],  # Lower bounds
                                    [b[1] for b in bounds_list]   # Upper bounds
                                ),
                                loss='soft_l1',
                                f_scale=0.5,
                                max_nfev=1000,
                                verbose=0
                            )

                            if res.success:
                                optimization_success = True
                                optimization_method = "least_squares (soft_l1) with priors"
                                st.success(f"✅ Optimization converged! Final cost: {res.cost:.8f}")
                            else:
                                st.warning(f"least_squares failed: {res.message}")
                        except Exception as e:
                            st.warning(f"least_squares crashed: {str(e)}")
                        
                        # Fallback to L-BFGS-B if least_squares fails
                        if not optimization_success:
                            st.info("Trying L-BFGS-B as fallback...")
                            try:
                                # Convert to scalar objective for minimize
                                def scalar_objective(params):
                                    residuals = lv_residual_function(params)
                                    return 0.5 * np.sum(residuals**2)
                                
                                res_fallback = minimize(
                                    scalar_objective,
                                    init_params,
                                    method="L-BFGS-B",
                                    bounds=bounds_list,
                                    options={'ftol': 1e-9, 'gtol': 1e-8, 'maxiter': 1000}
                                )
                                if res_fallback.success:
                                    # Convert minimize result to least_squares-like format
                                    class FallbackResult:
                                        def __init__(self, minimize_res):
                                            self.x = minimize_res.x
                                            self.success = minimize_res.success
                                            self.message = minimize_res.message
                                            self.nfev = minimize_res.nfev
                                            self.cost = minimize_res.fun
                                            
                                    res = FallbackResult(res_fallback)
                                    optimization_success = True
                                    optimization_method = "L-BFGS-B (fallback)"
                                else:
                                    st.warning(f"L-BFGS-B fallback failed: {res_fallback.message}")
                            except Exception as e:
                                st.warning(f"L-BFGS-B fallback crashed: {str(e)}")
                        
                        if not optimization_success:
                            st.error("❌ All optimization strategies failed. Please check:")
                            st.error("1. Data quality - ensure component tracks have sufficient signal")
                            st.error("2. Constraints - try relaxing sign constraints")
                            st.error("3. Data format - ensure Format 2/3 with component series")
                            st.stop()

                        # Extract fitted parameters from optimization result
                        # New structure: [μ₀, a₀₀, μ₁, a₁₁, ..., α₀₁, α₁₀, α₀₂, α₂₀, ...]
                        
                        # Extract fitted monoculture parameters (μ_i, a_ii)
                        fitted_mu_vals = np.zeros(n_species)
                        fitted_aii_vals = np.zeros(n_species)
                        
                        for i in range(n_species):
                            fitted_mu_vals[i] = res.x[2*i]      # μ_i
                            fitted_aii_vals[i] = res.x[2*i + 1] # a_ii
                        
                        # Extract fitted cross-interactions (α_ij, α_ji)
                        A_fit = np.zeros((n_species, n_species), dtype=float)
                        
                        # Set fitted diagonal elements
                        np.fill_diagonal(A_fit, fitted_aii_vals)
                        
                        # Fill cross-interactions from optimization result
                        alpha_start_idx = 2 * n_species  # Start after monoculture parameters
                        param_idx = alpha_start_idx
                        for i, j in pair_indices:
                            # Extract the two cross-interactions for this pair
                            alpha_ij = res.x[param_idx]      # i -> j interaction  
                            alpha_ji = res.x[param_idx + 1]  # j -> i interaction
                            
                            # Set in full matrix
                            A_fit[i, j] = alpha_ij
                            A_fit[j, i] = alpha_ji
                            
                            param_idx += 2
                        
                        # Update mu_vals to use fitted values instead of fixed monoculture values
                        mu_vals = fitted_mu_vals.copy()
                        
                        # Note: All parameters are now fitted simultaneously
                        st.session_state.A_fit = A_fit
                        st.session_state.mu_fit = mu_vals  # Store fitted growth rates
                        st.session_state.lv_fit_done = True
                        st.session_state.optimization_result = {
                            "method": optimization_method,
                            "success": res.success,
                            "message": getattr(res, 'message', 'Converged'),
                            "nfev": getattr(res, 'nfev', getattr(res, 'fun_evals', 'N/A')),
                            "cost": getattr(res, 'cost', getattr(res, 'fun', 'N/A'))
                        }
                        
                        # ChatGPT's consistency check: Quick multi-species validation
                        st.info("🔍 Running consistency check on assembled interaction matrix...")
                        try:
                            # Test simulation with full assembled matrix
                            test_sol = solve_ivp(
                                lv_ode,
                                (time_span[0], time_span[-1]),
                                x0_vals,
                                args=(mu_vals, A_fit),
                                t_eval=time_span,
                                method="RK45",
                                rtol=1e-6,
                                atol=1e-9
                            )
                            if test_sol.success:
                                st.success("✅ Full multi-species simulation successful with assembled matrix")
                            else:
                                st.warning("⚠️ Full simulation had issues - individual pairs may be more reliable")
                        except Exception as e:
                            st.warning(f"⚠️ Consistency check failed: {e}")

                        # Calculate parameter errors using Jacobian from least_squares
                        try:
                            if hasattr(res, 'jac') and res.jac is not None:
                                # Calculate covariance matrix from Jacobian
                                jac = res.jac
                                # Covariance matrix = inv(J^T J) for least squares
                                try:
                                    cov_matrix = np.linalg.inv(jac.T @ jac)
                                    param_errors_flat = np.sqrt(np.abs(np.diag(cov_matrix)))
                                    
                                    # Map to matrix format for all fitted parameters
                                    param_errors_matrix = np.full((n_species, n_species), np.nan)
                                    
                                    # Extract errors for monoculture parameters (μ_i, a_ii)
                                    for i in range(n_species):
                                        if 2*i < len(param_errors_flat):
                                            # μ_i error (not stored in matrix, but available in cov_matrix)
                                            pass
                                        if 2*i + 1 < len(param_errors_flat):
                                            # a_ii error
                                            param_errors_matrix[i, i] = param_errors_flat[2*i + 1]
                                    
                                    # Extract errors for cross-interactions
                                    alpha_start_idx = 2 * n_species
                                    param_idx = alpha_start_idx
                                    for i, j in pair_indices:
                                        if param_idx < len(param_errors_flat):
                                            param_errors_matrix[i, j] = param_errors_flat[param_idx]
                                        if param_idx + 1 < len(param_errors_flat):
                                            param_errors_matrix[j, i] = param_errors_flat[param_idx + 1]
                                        param_idx += 2
                                    
                                    st.session_state.param_errors = param_errors_matrix
                                    st.session_state.param_cov_matrix = cov_matrix
                                    st.session_state.param_errors_flat = param_errors_flat  # Store all parameter errors
                                    st.success("✅ Parameter errors calculated from optimization Jacobian")
                                except np.linalg.LinAlgError:
                                    st.warning("Could not calculate parameter errors - singular Jacobian")
                                    st.session_state.param_errors = np.full((n_species, n_species), np.nan)
                            else:
                                st.warning("No Jacobian available for error calculation")
                                st.session_state.param_errors = np.full((n_species, n_species), np.nan)
                        except Exception as e:
                            st.warning(f"Error calculation failed: {e}")
                            st.session_state.param_errors = np.full((n_species, n_species), np.nan)
                        
                        # Note: Diagonal errors are now calculated from the simultaneous optimization Jacobian
                        # No need to extract from individual fits since all parameters are fitted together

                        # ChatGPT approach: Simulate each pair independently with fitted parameters
                        st.info("Simulating fitted pairs for validation...")
                        pairwise_simulations = {}
                        
                        for i, j in pair_indices:
                            try:
                                # Get component data and measured initials
                                obs_i, obs_j, x0_pair_measured = get_pair_components_and_initials(i, j)
                                if obs_i is None or obs_j is None or x0_pair_measured is None:
                                    continue
                                
                                # Extract fitted 2x2 interaction matrix for this pair
                                A_pair = np.array([
                                    [A_fit[i, i], A_fit[i, j]],  # Species i interactions
                                    [A_fit[j, i], A_fit[j, j]]   # Species j interactions
                                ])
                                
                                # Use measured pairwise initials (ChatGPT approach)
                                x0_pair = np.clip(x0_pair_measured, 1e-9, None)
                                mu_pair = np.array([mu_vals[i], mu_vals[j]])
                                
                                # Define and solve 2-species LV system
                                def pairwise_lv_ode(t, x):
                                    x1, x2 = x
                                    dx1 = x1 * (mu_pair[0] + A_pair[0, 0] * x1 + A_pair[0, 1] * x2)
                                    dx2 = x2 * (mu_pair[1] + A_pair[1, 0] * x1 + A_pair[1, 1] * x2)
                                    return [dx1, dx2]
                                
                                max_step = np.diff(time_span).max() if len(time_span) > 1 else None
                                sol_pair = solve_ivp(
                                    pairwise_lv_ode,
                                    (time_span[0], time_span[-1]),
                                    x0_pair,
                                    t_eval=time_span,
                                    method="RK45",
                                    rtol=1e-6,
                                    atol=1e-9,
                                    max_step=max_step
                                )
                                
                                if sol_pair.success:
                                    pairwise_simulations[(i, j)] = sol_pair.y  # Shape: (2, n_time)
                                else:
                                    st.warning(f"Simulation failed for pair {species_cols[i]}+{species_cols[j]}")
                                    pairwise_simulations[(i, j)] = None
                                    
                            except Exception as e:
                                st.warning(f"Error simulating pair {species_cols[i]}+{species_cols[j]}: {e}")
                                pairwise_simulations[(i, j)] = None
                        
                        # Store results
                        st.session_state.lv_pairwise_simulations = pairwise_simulations
                        st.session_state.lv_fit_time = time_span
                        
                        # Also create full simulation for backward compatibility and comparison
                        try:
                            sol_full = solve_ivp(
                                lv_ode,
                                (time_span[0], time_span[-1]),
                                x0_vals,
                                args=(mu_vals, A_fit),
                                t_eval=time_span,
                                method="RK45",
                                rtol=1e-6,
                                atol=1e-9
                            )
                            if sol_full.success:
                                st.session_state.lv_species_sol = sol_full.y
                                st.success("✅ Full multi-species simulation also successful")
                            else:
                                st.session_state.lv_species_sol = None
                                st.warning("⚠️ Full simulation failed - using pairwise simulations only")
                        except Exception:
                            st.session_state.lv_species_sol = None

                        # Calculate metrics comparing fitted pairs to observed data
                        metrics = []
                        for i, j in pair_indices:
                            pair_name = f"{species_cols[i]}+{species_cols[j]}"
                            sim_pair = pairwise_simulations.get((i, j))
                            if sim_pair is None:
                                continue
                                
                            # Get observed component data
                            obs_i, obs_j, _ = get_pair_components_and_initials(i, j)
                            if obs_i is None or obs_j is None:
                                continue
                                
                            sim_i, sim_j = sim_pair[0], sim_pair[1]
                            
                            # Calculate metrics for both components combined
                            all_obs = np.concatenate([obs_i, obs_j])
                            all_sim = np.concatenate([sim_i, sim_j])
                            
                            # Filter finite values
                            mask = np.isfinite(all_obs) & np.isfinite(all_sim)
                            if mask.sum() >= 2:
                                obs_clean = all_obs[mask]
                                sim_clean = all_sim[mask]
                                
                                err = sim_clean - obs_clean
                                mse = float(np.mean(err ** 2))
                                rmse = float(np.sqrt(mse))
                                mae = float(np.mean(np.abs(err)))
                                
                                # R² calculation
                                ss_res = float(np.sum(err ** 2))
                                ss_tot = float(np.sum((obs_clean - np.mean(obs_clean)) ** 2))
                                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                                
                                # Pearson correlation
                                if np.std(obs_clean) > 0 and np.std(sim_clean) > 0:
                                    r = float(np.corrcoef(obs_clean, sim_clean)[0, 1])
                                else:
                                    r = np.nan
                            else:
                                mse = rmse = mae = r2 = r = np.nan
                                
                            metrics.append({
                                "Pair": pair_name,
                                "MSE": mse,
                                "RMSE": rmse,
                                "MAE": mae,
                                "R²": r2,
                                "Pearson r": r
                            })
                        
                        met_df = pd.DataFrame(metrics)
                        st.session_state.lv_pairwise_metrics = met_df

                # --- Fitted μ table (from LV fit) ---
                if "mu_fit" in st.session_state and st.session_state.mu_fit is not None:
                    st.markdown("#### Fitted Growth Rates (μ) from LV Fit")
                    try:
                        mu_lv = np.asarray(st.session_state.mu_fit, dtype=float)
                        rows = []
                        for i, sp in enumerate(species_cols):
                            mu_lv_i = float(mu_lv[i]) if i < len(mu_lv) and np.isfinite(mu_lv[i]) else np.nan
                            # Use current priors if available in this scope/session
                            mu_prior_i = np.nan
                            try:
                                if 'mu_priors' in locals() and i < len(mu_priors):
                                    mu_prior_i = float(mu_priors[i])
                            except Exception:
                                mu_prior_i = np.nan
                            # Monoculture μ (from individual fits) if available
                            mu_mono_i = np.nan
                            df_fit_local = st.session_state.get("df_fit")
                            if df_fit_local is not None:
                                try:
                                    if 'mu_val' in df_fit_local.columns:
                                        vals = df_fit_local.loc[df_fit_local["Species"] == sp, "mu_val"].values
                                    elif "Growth Rate (μ)" in df_fit_local.columns:
                                        vals = df_fit_local.loc[df_fit_local["Species"] == sp, "Growth Rate (μ)"].values
                                    else:
                                        vals = []
                                    if len(vals):
                                        mu_mono_i = float(vals[0])
                                except Exception:
                                    mu_mono_i = np.nan
                            rows.append({
                                "Species": sp,
                                "μ (LV fit)": mu_lv_i,
                                "μ prior": mu_prior_i,
                                "μ (monoculture)": mu_mono_i
                            })
                        mu_table = pd.DataFrame(rows)
                        st.dataframe(mu_table, use_container_width=True)
                        st.markdown(download_csv(mu_table, "lv_mu_values"), unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not render μ table: {e}")

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
                                        x=np.concatenate([lv_fit_time, lv_fit_time[::-1]]),
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
                    # Calculate error vs data for each species (mask invalid to avoid None/NaN floods)
                    lv_sol_df = pd.DataFrame(sim_species.T, columns=species_cols)
                    lv_sol_df["time"] = lv_fit_time
                    lv_sol_df = lv_sol_df[["time"] + species_cols]
                    err_rows = []
                    for i, sp in enumerate(species_cols):
                        obs = np.asarray(data_proc[sp].values, dtype=float)
                        sim = np.asarray(sim_species[i], dtype=float)
                        mask = np.isfinite(obs) & np.isfinite(sim)
                        if mask.sum() >= 2:
                            sd_err = float(np.nanstd(sim[mask] - obs[mask]))
                        else:
                            sd_err = np.nan
                        err_rows.append({"Species": sp, "SD Error (Sim - Data)": sd_err})
                    err_table = pd.DataFrame(err_rows)
                    st.dataframe(lv_sol_df.head(10), use_container_width=True)
                    st.dataframe(err_table, use_container_width=True)
                    st.caption("SD Error computed on overlapping valid points only. N/A means insufficient overlapping data.")

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
                    
                    # Use biological methodology: pairwise simulations instead of full species simulation
                    if "lv_pairwise_simulations" in st.session_state and st.session_state.lv_pairwise_simulations is not None:
                        pairwise_sims = st.session_state.lv_pairwise_simulations
                    else:
                        pairwise_sims = {}
                    
                    # Add the main curves on top
                    for idx, (i, j) in enumerate(pair_indices):
                            name = f"{species_cols[i]}+{species_cols[j]}"
                            color = colors[idx % len(colors)]
                            
                            # Get pairwise simulation for this pair
                            sim_pair = pairwise_sims.get((i, j))
                            
                            # Handle different data formats properly
                            obs_sum = None
                            sim_sum = None
                            
                            if data_format == 'format_1_total_pairs':
                                # Format 1: Direct pairwise sums
                                if name in data_proc.columns:
                                    obs_sum = data_proc[name].values
                                if sim_pair is not None:
                                    sim_sum = sim_pair[0] + sim_pair[1]  # Sum of 2 species from pairwise simulation
                            else:
                                # Format 2/3: Individual species counts from pairwise cultures
                                species_pair_indices_1based = tuple(sorted([i+1, j+1]))  # Convert to 1-based for data structure lookup
                                
                                # Find individual counts for this pair
                                pair_data = []
                                for col_name, col_info in data_structure.get('pairwise_combinations', {}).items():
                                    if (col_info['type'] == 'individual_count_in_pair' and 
                                        tuple(sorted(col_info['species_indices'])) == species_pair_indices_1based):
                                        pair_data.append(data_proc[col_name].values)
                                
                                # Sum the individual counts if available
                                if len(pair_data) >= 2:
                                    obs_sum = pair_data[0] + pair_data[1]
                                elif len(pair_data) == 1:
                                    # Only one species data available, use it as approximate
                                    obs_sum = pair_data[0] * 2  # Rough approximation
                                
                                # Get simulation sum from pairwise simulation
                                if sim_pair is not None:
                                    sim_sum = sim_pair[0] + sim_pair[1]
                            
                            # Add observed data if available
                            if obs_sum is not None:
                                # Observed data (no CI)
                                fig2.add_trace(go.Scatter(
                                    x=lv_fit_time, y=obs_sum,
                                    mode="markers+lines",
                                    name=f"Obs {name}",
                                    line=dict(color=color),
                                    marker=dict(size=4)
                                ))
                            
                            # Add simulated data if available
                            if sim_sum is not None:
                                # Simulated data (fitted curve on top of CI)
                                fig2.add_trace(go.Scatter(
                                    x=lv_fit_time, y=sim_sum,
                                    mode="lines",
                                    name=f"Sim {name}",
                                    line=dict(color=color, dash="dash", width=3)
                                ))
                            else:
                                # Fallback: use full species simulation if pairwise failed
                                if st.session_state.lv_species_sol is not None:
                                    sim_species = st.session_state.lv_species_sol
                                    sim_sum_fallback = sim_species[i] + sim_species[j]
                                    fig2.add_trace(go.Scatter(
                                        x=lv_fit_time, y=sim_sum_fallback,
                                        mode="lines",
                                        name=f"Sim {name} (fallback)",
                                        line=dict(color=color, dash="dot", width=2)
                                    ))
                    
                    fig2.update_layout(
                        title="Biological Pairwise Fitting: Independent 2-Species Simulations vs Observed Data",
                        xaxis_title="Time", yaxis_title="Population (OD)",
                        template="plotly_white"
                    )
                    if lv_log_scale:
                        fig2.update_yaxes(type="log")
                    st.plotly_chart(fig2, use_container_width=True)

                # === Individual Species Trajectories in Coculture ===
                st.markdown("#### Individual Species in Coculture: Observed vs Simulated")

                if "lv_pairwise_simulations" in st.session_state and st.session_state.lv_pairwise_simulations is not None:
                    pairwise_sims = st.session_state.lv_pairwise_simulations

                    # Create plots for each pair showing individual species trajectories
                    for pair_idx, (i, j) in enumerate(pair_indices):
                        pair_name = f"{species_cols[i]}+{species_cols[j]}"
                        sim_pair = pairwise_sims.get((i, j))

                        if sim_pair is not None:
                            # Create figure for this pair
                            fig_individual = go.Figure()

                            # Get the observed individual count data for this pair
                            obs_i, obs_j, _ = get_pair_components_and_initials(i, j)

                            # Add observed and simulated trajectories
                            color_i = colors[i % len(colors)]
                            color_j = colors[j % len(colors)]

                            # Species i trajectory
                            if obs_i is not None:
                                fig_individual.add_trace(go.Scatter(
                                    x=lv_fit_time, y=obs_i,
                                    mode="markers+lines",
                                    name=f"Obs {species_cols[i]}({pair_name})",
                                    line=dict(color=color_i),
                                    marker=dict(size=4)
                                ))

                            fig_individual.add_trace(go.Scatter(
                                x=lv_fit_time, y=sim_pair[0],
                                mode="lines",
                                name=f"Sim {species_cols[i]}({pair_name})",
                                line=dict(color=color_i, dash="dash", width=3)
                            ))

                            # Species j trajectory
                            if obs_j is not None:
                                fig_individual.add_trace(go.Scatter(
                                    x=lv_fit_time, y=obs_j,
                                    mode="markers+lines",
                                    name=f"Obs {species_cols[j]}({pair_name})",
                                    line=dict(color=color_j),
                                    marker=dict(size=4)
                                ))

                            fig_individual.add_trace(go.Scatter(
                                x=lv_fit_time, y=sim_pair[1],
                                mode="lines",
                                name=f"Sim {species_cols[j]}({pair_name})",
                                line=dict(color=color_j, dash="dash", width=3)
                            ))

                            fig_individual.update_layout(
                                title=f"Individual Species Trajectories: {pair_name} Coculture",
                                xaxis_title="Time",
                                yaxis_title="Population (OD)",
                                template="plotly_white",
                                height=400
                            )

                            if lv_log_scale:
                                fig_individual.update_yaxes(type="log")

                            st.plotly_chart(fig_individual, use_container_width=True)

                            # Add metrics for this pair
                            if obs_i is not None and obs_j is not None:
                                try:
                                    def calculate_r2(observed, simulated):
                                        if len(observed) != len(simulated):
                                            return np.nan
                                        ss_res = np.sum((observed - simulated) ** 2)
                                        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
                                        return 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                                    r2_i = calculate_r2(obs_i, sim_pair[0])
                                    r2_j = calculate_r2(obs_j, sim_pair[1])

                                    metrics_data = {
                                        "Species": [species_cols[i], species_cols[j]],
                                        "R²": [f"{r2_i:.3f}" if not np.isnan(r2_i) else "N/A",
                                               f"{r2_j:.3f}" if not np.isnan(r2_j) else "N/A"]
                                    }
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                except:
                                    pass

                # === Show Fitted α Matrix ===
                st.markdown("### Fitted interaction matrix (α)")
                
                # Show optimization summary
                if "optimization_result" in st.session_state:
                    opt_result = st.session_state.optimization_result
                    with st.expander("🔍 Optimization Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            # least_squares doesn't have method, show algorithm instead
                            method_name = getattr(opt_result, 'method', 'least_squares')
                            st.metric("Algorithm", method_name)
                        with col2:
                            # least_squares uses 'nfev' (function evaluations) instead of 'nit'
                            iterations = getattr(opt_result, 'nfev', getattr(opt_result, 'nit', 'N/A'))
                            st.metric("Function Evals", iterations)
                        with col3:
                            # For least_squares, cost = 0.5 * sum(residuals**2)
                            final_cost = getattr(opt_result, 'cost', getattr(opt_result, 'fun', 'N/A'))
                            if isinstance(final_cost, (int, float)):
                                st.metric("Final Cost", f"{final_cost:.8f}")
                            else:
                                st.metric("Final Cost", str(final_cost))
                        
                        # Show optimization status/message
                        status_msg = getattr(opt_result, 'message', getattr(opt_result, 'status', 'Complete'))
                        st.info(f"Optimization status: {status_msg}")
                
                # === Parameter Table with Values and Errors ===
                st.markdown("#### Parameter Table with Standard Errors")
                
                # Create parameter table similar to Model Fitting tab
                if hasattr(st.session_state, 'param_errors') and st.session_state.param_errors is not None:
                    # Determine which off-diagonal entries were actually estimated
                    data_format = st.session_state.get('data_format', 'format_1_total_pairs')
                    format2 = (data_format == 'format_2_focal_individual')
                    try:
                        estimated_pairs = set(tuple(sorted((i, j))) for (i, j) in pair_indices)
                    except Exception:
                        estimated_pairs = set()

                    param_table_data = []
                    for i in range(n_species):
                        for j in range(n_species):
                            param_name = f"α({species_cols[i]},{species_cols[j]})"
                            param_value = st.session_state.A_fit[i, j]
                            param_error = st.session_state.param_errors[i, j]
                            
                            # Override diagonals with individual fit SE/CI when available
                            if i == j and 'df_fit' in st.session_state and st.session_state.df_fit is not None:
                                try:
                                    row = st.session_state.df_fit[st.session_state.df_fit['Species'] == species_cols[i]]
                                    if not row.empty:
                                        # Use SE and CI from individual fit
                                        a11_se = row['a11_se'].iloc[0]
                                        a11_ci_low = row['a11_CI_low'].iloc[0]
                                        a11_ci_high = row['a11_CI_high'].iloc[0]
                                        if np.isfinite(a11_se) and a11_se > 0:
                                            param_error = a11_se
                                            ci_low = a11_ci_low
                                            ci_high = a11_ci_high
                                            value_str = f"{param_value:.8f} ± {param_error:.8f}"
                                            # Simple significance check
                                            t_stat = abs(param_value / param_error) if param_error else 0
                                            if t_stat > 2.58:
                                                significance = "***"
                                            elif t_stat > 1.96:
                                                significance = "**"
                                            elif t_stat > 1.645:
                                                significance = "*"
                                            else:
                                                significance = ""
                                            # Append row and continue
                                            is_negative = st.session_state.constraint_matrix_negative[i][j]
                                            is_positive = st.session_state.constraint_matrix_positive[i][j]
                                            constraint = "< 0" if (is_negative and not is_positive) else "> 0" if (is_positive and not is_negative) else "free"
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
                                            continue
                                except Exception:
                                    pass

                            # In Format 2, mark non-focal off-diagonals as not estimated (N/A)
                            is_offdiag = (i != j)
                            was_estimated = (not is_offdiag) or (tuple(sorted((i, j))) in estimated_pairs)
                            if format2 and is_offdiag and not was_estimated:
                                ci_low = np.nan
                                ci_high = np.nan
                                value_str = "N/A (no data in Format 2)"
                                significance = ""
                                param_error = np.nan
                            else:
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
                        st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.1. All parameters (μ_i, a_ii, α_ij) fitted simultaneously using robust least_squares optimization with soft_l1 loss and soft priors on monoculture parameters.")
                        if format2:
                            st.caption("Format 2: Only pairs involving focal species are estimated. Other off-diagonal α(i,j) are shown as N/A and were not fitted.")
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
                # For display: in Format 2, mask non-focal off-diagonals as NaN (not estimated)
                data_format = st.session_state.get('data_format', 'format_1_total_pairs')
                format2 = (data_format == 'format_2_focal_individual')
                try:
                    estimated_pairs = set(tuple(sorted((i, j))) for (i, j) in pair_indices)
                except Exception:
                    estimated_pairs = set()
                alpha_display = alpha_df.copy()
                if format2:
                    for i in range(n_species):
                        for j in range(n_species):
                            if i != j and tuple(sorted((i, j))) not in estimated_pairs:
                                alpha_display.iat[i, j] = np.nan
                
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
                    column_config = {col: st.column_config.NumberColumn(format="%.8f") for col in alpha_display.columns}
                    st.dataframe(alpha_display, use_container_width=True, column_config=column_config)
                    if format2:
                        st.caption("Blank cells indicate interactions not estimated in Format 2 (no focal-pair data).")
                with col2:
                    st.write("**Applied constraints:**")
                    st.dataframe(constraint_df, use_container_width=True)

                # === Show Pairwise Sum Fit Metrics ===
                st.markdown("### Pairwise‐Sum Fit Metrics")
                if (st.session_state.lv_pairwise_metrics is not None and 
                    not st.session_state.lv_pairwise_metrics.empty):
                    st.dataframe(st.session_state.lv_pairwise_metrics, use_container_width=True)
                    st.markdown("**Average across all pairs:**")
                    avg = st.session_state.lv_pairwise_metrics.mean(numeric_only=True).to_frame("Average").T
                    st.dataframe(avg, use_container_width=True)
                else:
                    st.warning("No pairwise metrics available. This may indicate an issue with the fitting process.")
                
                st.success("Lotka-Volterra full species fit completed successfully!")

            else:
                st.info("Press 'Run LV Full‐Species Pairwise Sum Fit' to perform fitting and see results.")




        with fit_tabs[5]:
            st.subheader("Manual Exploration: Species Fits and Pairwise LV Simulation")
            
            # Add log scale checkbox for LV plots
            lv_plots_log_scale = st.checkbox("Use logarithmic Y-axis for LV plots", value=False, key="lv_plots_log_scale")

            n_species = len(species_cols)
            time_span = df_avg["time"].values

            # Defaults for μ (prefer fitted values, then priors, then df_fit, else default)
            mu_default = []
            df_fit_local = st.session_state.get("df_fit")
            mu_fit_ss = st.session_state.get("mu_fit")
            priors_ss = st.session_state.get("lv_priors")
            for sp in species_cols:
                val = None
                # 1) Use latest fitted μ from LV if available
                if isinstance(mu_fit_ss, (list, np.ndarray)) and len(mu_fit_ss) == len(species_cols):
                    val = float(mu_fit_ss[species_cols.index(sp)])
                # 2) Else use saved priors
                elif priors_ss and isinstance(priors_ss.get("mu"), dict) and sp in priors_ss["mu"]:
                    try:
                        val = float(priors_ss["mu"][sp])
                    except Exception:
                        val = None
                # 3) Else use monoculture df_fit if present
                elif df_fit_local is not None:
                    mu_arr = df_fit_local.loc[df_fit_local["Species"] == sp, "mu_val"].values
                    if len(mu_arr) and np.isfinite(mu_arr[0]):
                        val = float(mu_arr[0])
                # 4) Fallback default
                if val is None or not np.isfinite(val):
                    val = 0.01
                mu_default.append(val)
            mu_default = np.array(mu_default, dtype=float)

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

            # Add manual refresh buttons for μ and α
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("#### Parameter Configuration")
            with col2:
                if st.button("🔄 Sync μ from LV Fit", help="Update μ with latest values from LV Biological Fit"):
                    mu_fit_ss = st.session_state.get("mu_fit")
                    if isinstance(mu_fit_ss, (list, np.ndarray)) and len(mu_fit_ss) == n_species:
                        st.session_state.manual_mu_inputs = np.array(mu_fit_ss, dtype=float).copy()
                        st.success("✅ μ synchronized with LV fit results!")
                    else:
                        st.warning("No LV μ found. Run LV Biological Fit first.")
            with col3:
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

            # --- INDIVIDUAL SPECIES LOGISTIC FIT (using fitted parameters from Model Fitting) ---
            st.markdown("### Individual Species Logistic Fit (with self-interactions)")

            fig_logistic = go.Figure()
            colors = px.colors.qualitative.Bold
            
            # Get fitted parameters from the individual fitting results
            df_fit = st.session_state.get('df_fit')
            if df_fit is not None:
                for i, sp in enumerate(species_cols):
                    # Get fitted parameters for this species
                    fit_row = df_fit[df_fit["Species"] == sp]
                    if not fit_row.empty:
                        x0_fit = fit_row["x0_val"].iloc[0]
                        mu_fit = fit_row["mu_val"].iloc[0]
                        a11_fit = fit_row["a11_val"].iloc[0]
                        
                        # Get observed data with NA handling
                        y_data = data_proc[sp].values
                        valid_mask = ~np.isnan(y_data)
                        valid_time = time_span[valid_mask]
                        valid_y = y_data[valid_mask]
                        
                        # Plot observed data (only valid points, gaps preserved)
                        fig_logistic.add_trace(go.Scatter(
                            x=valid_time, 
                            y=valid_y,
                            mode="markers+lines", 
                            name=f"{sp} data",
                            line=dict(color=colors[i % len(colors)]),
                            marker=dict(size=4),
                            connectgaps=False
                        ))
                        
                        # Plot fitted logistic curve if parameters are valid
                        if np.isfinite([x0_fit, mu_fit, a11_fit]).all():
                            fitted_curve = logistic_model(time_span, x0_fit, mu_fit, a11_fit)
                            fig_logistic.add_trace(go.Scatter(
                                x=time_span, 
                                y=fitted_curve,
                                mode="lines", 
                                name=f"{sp} logistic fit",
                                line=dict(color=colors[i % len(colors)], dash="dash", width=2),
                                hovertemplate=f"<b>{sp}</b><br>" +
                                            f"μ={mu_fit:.4f}<br>" +
                                            f"a₁₁={a11_fit:.4f}<br>" +
                                            f"x₀={x0_fit:.4f}<br>" +
                                            "x=%{x}<br>y=%{y}<extra></extra>"
                            ))
                    else:
                        # Fallback: show observed data only
                        y_data = data_proc[sp].values
                        valid_mask = ~np.isnan(y_data)
                        valid_time = time_span[valid_mask]
                        valid_y = y_data[valid_mask]
                        
                        fig_logistic.add_trace(go.Scatter(
                            x=valid_time, 
                            y=valid_y,
                            mode="markers+lines", 
                            name=f"{sp} data (no fit)",
                            line=dict(color=colors[i % len(colors)]),
                            connectgaps=False
                        ))
            else:
                st.warning("No individual fit results found. Please run Model Fitting first.")
                
            fig_logistic.update_layout(
                title="Individual Species Modified Growth Fit (dX/dt = μ(X + a₁₁X))",
                xaxis_title="Time",
                yaxis_title="Population (OD)",
                template="plotly_white",
                height=500
            )
            if lv_plots_log_scale:
                fig_logistic.update_yaxes(type="log")
            st.plotly_chart(fig_logistic, use_container_width=True)
            
            # Show fitted parameters summary
            if df_fit is not None:
                st.markdown("**Fitted Parameters Summary:**")
                param_summary = df_fit[["Species", "Growth Rate (μ)", "Self-Interaction (a₁₁)", "R²"]]
                st.dataframe(param_summary, use_container_width=True)

            # --- PAIRWISE SUMS, LV SIMULATION (using μ and α) ---
            st.markdown("### Lotka-Volterra Pairwise Simulation (using μ and α)")

            from itertools import combinations
            
            # FIXED: Use the same pair logic as the biological fitting
            data_format = st.session_state.get('data_format', 'format_1_total_pairs')
            data_structure = st.session_state.get('data_structure', {})
            
            if data_format == 'format_2_focal_individual':
                # For Format 2, only include pairs involving the focal species
                focal_species = st.session_state.get('focal_species', 'Species 1')
                focal_index = int(focal_species.split()[-1]) - 1  # Convert "Species 1" to index 0
                
                # Only create pairs involving the focal species
                pair_indices = []
                for j in range(n_species):
                    if j != focal_index:
                        # Create pair (focal_index, j) in sorted order for consistency
                        if focal_index < j:
                            pair_indices.append((focal_index, j))
                        else:
                            pair_indices.append((j, focal_index))
                
                st.info(f"🎯 **Format 2 - Focal Species {focal_species}**: Showing {len(pair_indices)} pairs involving focal species")
                for i, j in pair_indices:
                    st.write(f"• {species_cols[i]} + {species_cols[j]}")
            else:
                # For other formats, use all possible pairs
                pair_indices = list(combinations(range(n_species), 2))
                st.info(f"📊 **{data_format.replace('_', ' ').title()}**: Showing {len(pair_indices)} pairwise combinations")
            
            # Smart initial condition estimation
            x0_smart, x0_info = estimate_initial_conditions_smart(data_proc, species_cols, time_span)
            
            # Display initial condition handling info
            with st.expander("🔍 Manual LV Simulation - Initial Condition Handling", expanded=False):
                st.write("**How initial conditions were determined:**")
                for info in x0_info:
                    st.write(f"• {info}")
            
            # Get initial values using smart estimation
            x0_vals = x0_smart
            
            # Critical safety check: ensure all initial values are finite
            if not np.all(np.isfinite(x0_vals)):
                st.error("⚠️ **Initial values contain NaN/infinite values. Cannot proceed with simulation.**")
                nan_species = [species_cols[i] for i in range(len(x0_vals)) if not np.isfinite(x0_vals[i])]
                st.write(f"**Problematic species**: {', '.join(nan_species)}")
                st.info("💡 **Solution**: These species may have missing data at the first time point. Consider using data from a later time point or filling missing values.")
                st.stop()
            
            # Additional validation: ensure all values are positive
            if not np.all(x0_vals > 0):
                st.warning("⚠️ **Some initial values are zero or negative. Setting to small positive values.**")
                x0_vals = np.maximum(x0_vals, 1e-6)
                st.write(f"**Adjusted initial values**: {x0_vals}")

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
            
            # Handle different data formats properly
            for idx, (i, j) in enumerate(pair_indices):
                name = f"{species_cols[i]}+{species_cols[j]}"
                
                # Try to get pairwise data based on format
                obs_sum = None
                if data_format == 'format_1_total_pairs':
                    # Format 1: Direct pairwise sums
                    if name in data_proc.columns:
                        obs_sum = data_proc[name].values
                else:
                    # Format 2/3: Sum individual species counts from pairwise cultures
                    species_pair_indices_1based = tuple(sorted([i+1, j+1]))  # Convert to 1-based for data structure lookup
                    
                    # Find individual counts for this pair
                    pair_data = []
                    for col_name, col_info in data_structure.get('pairwise_combinations', {}).items():
                        if (col_info['type'] == 'individual_count_in_pair' and 
                            tuple(sorted(col_info['species_indices'])) == species_pair_indices_1based):
                            pair_data.append(data_proc[col_name].values)
                    
                    # Sum the individual counts if available
                    if len(pair_data) >= 2:
                        obs_sum = pair_data[0] + pair_data[1]
                    elif len(pair_data) == 1:
                        # Only one species data available, use it as approximate
                        obs_sum = pair_data[0] * 2  # Rough approximation
                
                # Skip this pair if no data available
                if obs_sum is None:
                    st.warning(f"⚠️ No data available for pair {name}")
                    continue
                
                sim_sum = sol.y[i] + sol.y[j]
                color = colors[idx % len(colors)]
                
                # Filter out NaN values for observed data and plot as solid line
                valid_mask = ~np.isnan(obs_sum)
                valid_time = time_span[valid_mask]
                valid_obs = obs_sum[valid_mask]
                
                fig_pairwise.add_trace(go.Scatter(
                    x=valid_time, y=valid_obs,
                    mode="lines+markers", name=f"Obs {name}",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    connectgaps=False
                ))
                fig_pairwise.add_trace(go.Scatter(
                    x=time_span, y=sim_sum,
                    mode="lines", name=f"LV sim {name}",
                    line=dict(color=color, dash="dash", width=2)
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
                    st.rerun

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
                # Smart initial condition estimation
                x0_smart, x0_info = estimate_initial_conditions_smart(data_proc, species_cols, time_span)
                
                # Display initial condition handling info for phase plane
                with st.expander("🔍 Phase Plane - Initial Condition Handling", expanded=False):
                    st.write("**How initial conditions were determined:**")
                    for info in x0_info:
                        st.write(f"• {info}")
                
                # Get initial values using smart estimation
                x0_vals = x0_smart
                
                # Critical safety check: ensure all initial values are finite
                if not np.all(np.isfinite(x0_vals)):
                    st.error("⚠️ **Initial values contain NaN/infinite values. Cannot create quiver plot.**")
                    nan_species = [species_cols[k] for k in range(len(x0_vals)) if not np.isfinite(x0_vals[k])]
                    st.write(f"**Problematic species**: {', '.join(nan_species)}")
                    st.info("💡 **Solution**: These species may have missing data at the first time point.")
                    st.stop()
                
                # Additional validation: ensure all values are positive
                if not np.all(x0_vals > 0):
                    st.warning("⚠️ **Some initial values are zero or negative. Setting to small positive values for quiver plot.**")
                    x0_vals = np.maximum(x0_vals, 1e-6)
                
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
        
        # ========== Full Community Validation Section ==========
        st.markdown("---")
        st.markdown("### 🌐 Full Community Validation")
        st.markdown("Compare Lotka-Volterra model predictions with actual full community data")
        
        # Check if full community data is available
        full_community_cols = st.session_state.get('full_community_cols', [])
        if full_community_cols and len(full_community_cols) > 0:
            st.success(f"✅ Full community data found: {len(full_community_cols)} species tracked")
            
            # Check if we have fitted α matrix
            if "A_fit" in st.session_state and st.session_state.A_fit is not None:
                A_matrix = st.session_state.A_fit
                
                # Settings for full community simulation
                col1, col2 = st.columns(2)
                with col1:
                    sim_time_points = st.slider("Simulation time points", 50, 500, 200, 
                                               help="Number of time points for simulation")
                with col2:
                    community_log_scale = st.checkbox("Log scale for community plot", 
                                                     value=lv_plots_log_scale,
                                                     help="Use logarithmic scale for better visualization")
                
                try:
                    # Ensure all arrays are float64 to avoid casting errors
                    mu_inputs_float = np.array(mu_inputs, dtype=np.float64)
                    A_matrix_float = np.array(A_matrix, dtype=np.float64)
                    x0_vals_float = np.array(x0_vals, dtype=np.float64)
                    time_span_float = np.array(time_span, dtype=np.float64)
                    
                    # Simulate full multi-species system using fitted parameters
                    sol_full = solve_ivp(
                        lambda t, x: x * (mu_inputs_float + A_matrix_float @ x),
                        (time_span_float[0], time_span_float[-1]),
                        x0_vals_float,
                        t_eval=time_span_float,
                        method="RK45"
                    )
                    
                    if sol_full.success:
                        st.success("✅ Full community simulation successful!")
                        
                        # Create comparison plot
                        fig_community = go.Figure()
                        colors = px.colors.qualitative.Set1
                        
                        # Plot observed vs simulated for each species in full community
                        for i, col in enumerate(full_community_cols):
                            # Observed data from full community
                            obs_data = data_proc[col].values
                            
                            # Filter out NaN values for observed data
                            valid_mask = ~np.isnan(obs_data)
                            valid_time = time_span_float[valid_mask]
                            valid_obs = obs_data[valid_mask]
                            
                            # Simulated data for this species
                            sim_data = sol_full.y[i]
                            
                            species_name = col.replace("_community", "").upper()
                            color = colors[i % len(colors)]
                            
                            # Add observed data
                            if len(valid_obs) > 0:
                                fig_community.add_trace(go.Scatter(
                                    x=valid_time, y=valid_obs,
                                    mode="lines+markers", name=f"Obs {species_name}",
                                    line=dict(color=color, width=2),
                                    marker=dict(size=4),
                                    connectgaps=False
                                ))
                            
                            # Add simulated data
                            fig_community.add_trace(go.Scatter(
                                x=time_span_float, y=sim_data,
                                mode="lines", name=f"LV Pred {species_name}",
                                line=dict(color=color, dash="dash", width=3)
                            ))
                        
                        # Add total community comparison
                        obs_total = np.zeros_like(time_span_float)
                        valid_total_mask = np.ones_like(time_span_float, dtype=bool)
                        
                        for col in full_community_cols:
                            obs_data = data_proc[col].values
                            obs_total += np.nan_to_num(obs_data, 0)
                            valid_total_mask &= ~np.isnan(obs_data)
                        
                        sim_total = np.sum(sol_full.y, axis=0)
                        
                        # Filter total observed data
                        if valid_total_mask.sum() > 0:
                            valid_total_time = time_span_float[valid_total_mask]
                            valid_total_obs = obs_total[valid_total_mask]
                            
                            fig_community.add_trace(go.Scatter(
                                x=valid_total_time, y=valid_total_obs,
                                mode="lines+markers", name="Obs Total Community",
                                line=dict(color="black", width=4),
                                marker=dict(size=6),
                                connectgaps=False
                            ))
                        
                        fig_community.add_trace(go.Scatter(
                            x=time_span_float, y=sim_total,
                            mode="lines", name="LV Pred Total Community",
                            line=dict(color="black", dash="dash", width=4)
                        ))
                        
                        fig_community.update_layout(
                            title="Full Community: Lotka-Volterra Predictions vs Observed Data",
                            xaxis_title="Time",
                            yaxis_title="Population (OD)",
                            template="plotly_white",
                            height=600,
                            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top")
                        )
                        
                        if community_log_scale:
                            fig_community.update_yaxes(type="log")
                        
                        st.plotly_chart(fig_community, use_container_width=True)
                        
                        # Calculate prediction accuracy metrics
                        st.markdown("#### 📊 Full Community Prediction Accuracy")
                        
                        accuracy_data = []
                        for i, col in enumerate(full_community_cols):
                            obs = data_proc[col].values
                            sim = sol_full.y[i]
                            
                            # Calculate metrics on overlapping valid points
                            valid_mask = ~np.isnan(obs)
                            if valid_mask.sum() >= 2:
                                obs_valid = obs[valid_mask]
                                sim_valid = sim[valid_mask]
                                
                                # Calculate accuracy metrics
                                mse = np.mean((sim_valid - obs_valid) ** 2)
                                rmse = np.sqrt(mse)
                                mae = np.mean(np.abs(sim_valid - obs_valid))
                                
                                # R-squared
                                ss_res = np.sum((obs_valid - sim_valid) ** 2)
                                ss_tot = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
                                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                                
                                # Correlation
                                if np.std(obs_valid) > 0 and np.std(sim_valid) > 0:
                                    corr = np.corrcoef(obs_valid, sim_valid)[0, 1]
                                else:
                                    corr = np.nan
                                
                                # Prediction quality assessment
                                if r2 > 0.8:
                                    quality = "Excellent"
                                elif r2 > 0.6:
                                    quality = "Good"
                                elif r2 > 0.3:
                                    quality = "Moderate"
                                else:
                                    quality = "Poor"
                                
                                accuracy_data.append({
                                    "Species": col.replace("_community", "").upper(),
                                    "R²": f"{r2:.4f}" if not np.isnan(r2) else "N/A",
                                    "RMSE": f"{rmse:.4f}",
                                    "MAE": f"{mae:.4f}",
                                    "Correlation": f"{corr:.4f}" if not np.isnan(corr) else "N/A",
                                    "Quality": quality
                                })
                        
                        if accuracy_data:
                            accuracy_df = pd.DataFrame(accuracy_data)
                            st.dataframe(accuracy_df, use_container_width=True)
                            
                            # Overall assessment
                            valid_r2_values = [float(row["R²"]) for row in accuracy_data if row["R²"] != "N/A"]
                            if valid_r2_values:
                                avg_r2 = np.mean(valid_r2_values)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average R²", f"{avg_r2:.3f}")
                                with col2:
                                    valid_corr_values = [float(row["Correlation"]) for row in accuracy_data if row["Correlation"] != "N/A"]
                                    if valid_corr_values:
                                        avg_corr = np.mean(valid_corr_values)
                                        st.metric("Average Correlation", f"{avg_corr:.3f}")
                                with col3:
                                    excellent_count = sum(1 for row in accuracy_data if row["Quality"] == "Excellent")
                                    st.metric("Excellent Predictions", f"{excellent_count}/{len(accuracy_data)}")
                                
                                # Interpretation
                                if avg_r2 > 0.8:
                                    st.success("🎯 **Excellent prediction**: LV model captures full community dynamics very well!")
                                    st.markdown("The pairwise-derived parameters successfully predict full community behavior.")
                                elif avg_r2 > 0.6:
                                    st.info("👍 **Good prediction**: LV model reasonably captures community dynamics")
                                    st.markdown("The model works well, with minor discrepancies likely due to measurement noise or higher-order interactions.")
                                elif avg_r2 > 0.3:
                                    st.warning("⚠️ **Moderate prediction**: Some discrepancies between model and observations")
                                    st.markdown("Consider checking for non-linear interactions or environmental factors not captured by the LV model.")
                                else:
                                    st.error("❌ **Poor prediction**: Significant mismatch - check model assumptions")
                                    st.markdown("The pairwise LV model may not be suitable for this system. Consider higher-order interaction models.")
                            else:
                                st.warning("No valid R² values could be calculated")
                        else:
                            st.warning("No valid full community data found for accuracy calculation")
                    
                    else:
                        st.error("❌ Full community simulation failed")
                        st.write("Simulation error details:", sol_full.message if hasattr(sol_full, 'message') else "Unknown error")
                        
                except Exception as e:
                    st.error(f"Error in full community simulation: {e}")
                    st.markdown("**Possible issues:**")
                    st.markdown("- Initial conditions may be too extreme")
                    st.markdown("- Interaction matrix may contain unstable parameters")
                    st.markdown("- Time span may be too long for the system")
            else:
                st.warning("⚠️ No fitted LV parameters available. Run LV Biological Fit first.")
                
        else:
            st.info("ℹ️ **No full community data available**")
            st.markdown("""
            **Full community comparison requires:**
            - Individual species counts in full community culture
            - Data format with community measurements (e.g., x1_community, x2_community)
            - Completed LV parameter fitting from pairwise data
            
            **What this would show:**
            - LV model prediction for all species growing together
            - Comparison with actual full community measurements  
            - Prediction accuracy metrics (R², correlation, RMSE)
            - Validation of pairwise-derived interaction parameters
            - Assessment of higher-order interaction effects
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

