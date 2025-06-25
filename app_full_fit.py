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
    page_title="Cofit Dashboard",
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
            return pd.read_excel(file, engine='xlrd', header=None)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file, engine='openpyxl', header=None)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel (.xls/.xlsx) file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def ode_model(t, x0, mu):
    def ode(t, x):
        return mu * x
    sol = solve_ivp(ode, (t[0], t[-1]), [x0], t_eval=t, method='RK45')
    return sol.y[0]

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

st.markdown('<h1 class="main-header">Cofit Dashboard</h1>', unsafe_allow_html=True)
main_tabs = st.tabs(["Data Upload", "Data Analysis", "Model Fitting", "About"])

# ========== Data Upload Tab ==========
with main_tabs[0]:
    st.markdown('<h2 class="section-header">Upload and Process Data</h2>', unsafe_allow_html=True)
    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        1. Select number of replicate files  
        2. Upload each file (CSV or Excel format)  
        3. **See your raw data displayed under each upload**  
        4. Specify the number of species  
        5. Click "Process Data" to continue
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
        with st.form(key='species_definition_form'):
            st.subheader("Configure Data Processing")
            if 'confirmed_species_count_val' not in st.session_state:
                st.session_state.confirmed_species_count_val = 1
            col1, col2 = st.columns(2)
            with col1:
                species_count_from_form = st.number_input(
                    "Number of species in each file", 
                    min_value=1, 
                    max_value=20, 
                    step=1,
                    value=st.session_state.confirmed_species_count_val,
                    key="species_count_input_widget"
                )
            with col2:
                subtract_bg_form = st.checkbox("Subtract background from data?", value=False)
            process_data_button = st.form_submit_button("Process Data", use_container_width=True)
            if process_data_button:
                with st.spinner("Processing data..."):
                    st.session_state.confirmed_species_count_val = species_count_from_form
                    dfs_check = []
                    for idx, df in enumerate(dfs):
                        if df is None:
                            st.error(f"Failed to read file {idx+1}")
                            break
                        min_cols = 1 + species_count_from_form + ((species_count_from_form*(species_count_from_form-1))//2) + 1
                        if df.shape[1] < min_cols:
                            st.error(f"File {idx+1} has fewer columns ({df.shape[1]}) than expected minimum ({min_cols})")
                            break
                        dfs_check.append(df)
                    if len(dfs_check) == int(num_replicates):
                        times = [df.iloc[:, 0] for df in dfs_check]
                        time_ref = times[0]
                        if not all(time_ref.equals(t) for t in times[1:]):
                            st.error("Time columns do not match across all replicate files.")
                        else:
                            data_arrays = [df.iloc[:, 1:].to_numpy(dtype=float) for df in dfs_check]
                            avg_data = np.mean(data_arrays, axis=0)
                            df_avg = pd.DataFrame(np.hstack([time_ref.values.reshape(-1,1), avg_data]))
                            species_cols = [f"x{i+1}" for i in range(species_count_from_form)]
                            pairwise_cols = []
                            for i in range(species_count_from_form):
                                for j in range(i+1, species_count_from_form):
                                    pairwise_cols.append(f"x{i+1}+x{j+1}")
                            col_names = ["time"] + species_cols + pairwise_cols + ["background_avg"]
                            if len(col_names) != df_avg.shape[1]:
                                total_cols = df_avg.shape[1]
                                correct_species = int((-1 + np.sqrt(1 + 8*(total_cols - 2))) / 2)
                                st.warning(f"""
                                ⚠️ **Column count mismatch!** 
                                The species count you specified ({species_count_from_form}) doesn't match your data structure.
                                Based on your data, the correct number of species is likely **{correct_species}**.
                                Please try again with {correct_species} species.
                                """)
                                st.session_state.confirmed_species_count_val = correct_species                            
                            else:
                                df_avg.columns = col_names
                                st.session_state.df_avg = df_avg
                                st.session_state.species_count = species_count_from_form
                                st.session_state.subtract_bg = subtract_bg_form
                                st.session_state.species_cols = species_cols
                                st.session_state.pairwise_cols = pairwise_cols
                                st.success("✅ Data processed successfully! Go to the Data Analysis section.")
                                st.balloons()

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
        fit_indices = np.where((time_data_full >= time_min) & (time_data_full <= time_max))[0]
        time_data = time_data_full[fit_indices]
        with st.spinner("Running model fitting..."):
            fit_results = []
            for species in species_cols:
                y_full = data_no_time[species].values
                y_data = y_full[fit_indices]
                try:
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
                    st.error(f"Fit failed for {species}: {e}")
            df_fit = pd.DataFrame(fit_results)
            st.session_state.df_fit = df_fit        

        st.subheader("Model Fitting Results")
        fit_tabs = st.tabs(["Parameter Table", "Fitted Curves", "Growth Rate Comparison", "Lotka-Volterra Pairwise", "Lotka-Volterra Fit", "LV Sum-Based Fit" ])
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

                # Fit and confidence bands (if not failed)
                if not np.isnan(x0_fit) and not np.isnan(mu_fit):
                    fitted_curve = ode_model(time_data, x0_fit, mu_fit)
                    fitted_curve_lower = ode_model(time_data, x0_fit, mu_CI_low)
                    fitted_curve_upper = ode_model(time_data, x0_fit, mu_CI_high)
                    color_hex = colors[idx % len(colors)]
                    # Convert hex to rgba for proper opacity handling
                    if color_hex.startswith('#'):
                        r = int(color_hex[1:3], 16)
                        g = int(color_hex[3:5], 16)
                        b = int(color_hex[5:7], 16)
                        rgba_color = f'rgba({r},{g},{b},0.2)'
                    else:
                        rgba_color = f'rgba(100,100,100,0.2)'
                    fig_fit.add_trace(go.Scatter(
                        x=time_data,
                        y=fitted_curve,
                        mode='lines',
                        name=f"{species} fit",
                        line=dict(color=color_hex, dash='dash', width=2)
                    ))
                    fig_fit.add_trace(go.Scatter(
                        x=np.concatenate([time_data, time_data[::-1]]),
                        y=np.concatenate([fitted_curve_lower, fitted_curve_upper[::-1]]),
                        fill='toself',
                        fillcolor=rgba_color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        name=f"{species} 95% CI"
                    ))
                    fig_fit.add_vrect(
                        x0=time_min, x1=time_max,
                        fillcolor="gray", opacity=0.07,
                        layer="below", line_width=0,
                    )
            fig_fit.update_layout(
                title="Individual Species Data with ODE Fit and 95% Confidence Interval",
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
            st.subheader("Lotka-Volterra Fit (Sum-based, using monoculture μ)")
            st.markdown("""
            - Select a co-culture pair (e.g., x1+x2).
            - Adjust α values manually or use **Auto-fit α's**.
            - The interaction matrix below shows the current α values.
            """)

            pair_options = [col for col in pairwise_cols]
            pair_selected = st.selectbox("Select co-culture pair:", pair_options)
            s1, s2 = [int(s) for s in pair_selected.lower().replace("x", "").split("+")]

            try:
                mu_1 = float(df_fit.loc[df_fit['Species'] == f'x{s1}', 'mu_val'])
                mu_2 = float(df_fit.loc[df_fit['Species'] == f'x{s2}', 'mu_val'])
                x1_0 = float(df_avg[f'x{s1}'].iloc[0])
                x2_0 = float(df_avg[f'x{s2}'].iloc[0])
            except Exception as e:
                st.warning("Could not retrieve initial values or growth rates from fits. Make sure monocultures were successfully fitted.")
                st.stop()

            fit_key = f"lv_fit_{pair_selected}"
            if fit_key not in st.session_state:
                st.session_state[fit_key] = {
                    "alpha_11": 0.0,
                    "alpha_12": 0.0,
                    "alpha_21": 0.0,
                    "alpha_22": 0.0,
                    "fitted": False
                }

            col1, col2 = st.columns(2)
            with col1:
                alpha_12 = st.number_input(f"α₁₂ (effect of x{s2} on x{s1})", value=st.session_state[fit_key]["alpha_12"], step=0.05, format="%.3f", key=f"alpha_12_{pair_selected}")
                alpha_11 = st.number_input(f"α₁₁ (self-interaction for x{s1})", value=st.session_state[fit_key]["alpha_11"], step=0.05, format="%.3f", key=f"alpha_11_{pair_selected}")
            with col2:
                alpha_21 = st.number_input(f"α₂₁ (effect of x{s1} on x{s2})", value=st.session_state[fit_key]["alpha_21"], step=0.05, format="%.3f", key=f"alpha_21_{pair_selected}")
                alpha_22 = st.number_input(f"α₂₂ (self-interaction for x{s2})", value=st.session_state[fit_key]["alpha_22"], step=0.05, format="%.3f", key=f"alpha_22_{pair_selected}")

            auto_fit = st.button("Auto-fit α's to minimize error", key=f"autofit_{pair_selected}")

            time_span = df_avg["time"].values
            observed_sum = df_avg[pair_selected].values

            if auto_fit:
                init_params = [alpha_11, alpha_12, alpha_21, alpha_22]
                bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
                res = minimize(
                    mse_obj, init_params, bounds=bounds, method='L-BFGS-B',
                    args=(mu_1, mu_2, x1_0, x2_0, time_span, observed_sum)
                )
                if res.success:
                    alpha_11, alpha_12, alpha_21, alpha_22 = res.x
                    st.success("Auto-fit complete!")
                    st.session_state[fit_key] = {
                        "alpha_11": float(alpha_11),
                        "alpha_12": float(alpha_12),
                        "alpha_21": float(alpha_21),
                        "alpha_22": float(alpha_22),
                        "fitted": True
                    }
                else:
                    st.warning("Auto-fit failed. Try different initial values or check your data.")

            # --- LV Simulation for plotting with current α's ---
            sim_x1, sim_x2, sim_sum = lv_sim(mu_1, mu_2, alpha_11, alpha_12, alpha_21, alpha_22, x1_0, x2_0, time_span)

            fig_lv = go.Figure()
            fig_lv.add_trace(go.Scatter(
                x=time_span,
                y=observed_sum,
                mode='markers+lines',
                name=f"Observed {pair_selected}",
                line=dict(color="black", width=3)
            ))
            fig_lv.add_trace(go.Scatter(
                x=time_span,
                y=sim_sum,
                mode='lines',
                name=f"LV Simulated ({pair_selected})",
                line=dict(color="red", dash='dash')
            ))
            fig_lv.add_trace(go.Scatter(
                x=time_span,
                y=sim_x1,
                mode='lines',
                name=f"LV Simulated x{s1}",
                line=dict(color="blue", dash='dot')
            ))
            fig_lv.add_trace(go.Scatter(
                x=time_span,
                y=sim_x2,
                mode='lines',
                name=f"LV Simulated x{s2}",
                line=dict(color="green", dash='dot')
            ))

            fig_lv.update_layout(
                title=f"Lotka-Volterra Fit: {pair_selected} (Sum-based)",
                xaxis_title="Time",
                yaxis_title="Population (OD)",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_lv, use_container_width=True)

            mse = np.mean((sim_sum - observed_sum) ** 2)
            st.write(f"**Mean squared error (MSE) between simulated and observed sum:** {mse:.4f}")

            st.write("### Interaction Matrix for this Pair")
            matrix_df = pd.DataFrame(
                [[alpha_11, alpha_12],
                [alpha_21, alpha_22]],
                index=[f'x{s1}', f'x{s2}'],
                columns=[f'x{s1}', f'x{s2}']
            )
            st.dataframe(matrix_df, use_container_width=True)

            st.write("### Parameters Used")
            param_df = pd.DataFrame({
                "Parameter": [f"μ₁ (r₁) [{f'x{s1}'}]", f"μ₂ (r₂) [{f'x{s2}'}]", f"α₁₁ [{f'x{s1}'}]", f"α₁₂ [{f'x{s1}'}←{f'x{s2}'}]", f"α₂₁ [{f'x{s2}'}←{f'x{s1}'}]", f"α₂₂ [{f'x{s2}'}]"],
                "Value": [mu_1, mu_2, alpha_11, alpha_12, alpha_21, alpha_22]
            })
            st.dataframe(param_df, use_container_width=True)
        
        with fit_tabs[5]:
            st.subheader("Lotka-Volterra Full Species Fit")

            df_avg = st.session_state.df_avg
            species_cols = st.session_state.species_cols
            df_fit = st.session_state.get('df_fit', None)

            if df_fit is None:
                st.warning("Please run individual species fitting first!")
                st.stop()

            # Extract mu and x0 values from individual fits and data
            mu_vals = []
            x0_vals = []
            for sp in species_cols:
                mu_val = df_fit.loc[df_fit['Species'] == sp, 'mu_val'].values
                x0_val = df_avg[sp].iloc[0]
                if len(mu_val) == 0 or np.isnan(mu_val[0]):
                    st.warning(f"Missing mu value for {sp}. Fit individual species first.")
                    st.stop()
                mu_vals.append(mu_val[0])
                x0_vals.append(x0_val)
            mu_vals = np.array(mu_vals)
            x0_vals = np.array(x0_vals)

            time_span = df_avg["time"].values
            observed_data = df_avg[species_cols].to_numpy()  # shape (times, species)
            n_species = len(species_cols)

            st.markdown("""
            This tab fits the Lotka-Volterra model simultaneously to all species using fixed growth rates (μ) from monoculture fits.
            
            It estimates the full interaction matrix (α) by minimizing the difference between model simulation and observed data.
            """)

            # Define ODE function
            def lv_ode_all_species(t, x, mu, alpha):
                dxdt = np.zeros_like(x)
                for i in range(n_species):
                    interaction = np.dot(alpha[i, :], x)
                    dxdt[i] = x[i] * (mu[i] + interaction)
                return dxdt

            # Initial interaction matrix: zeros (no interaction)
            init_alpha = np.zeros((n_species, n_species))

            # Simulate ODE with current alpha (for initial plot)
            sol_init = solve_ivp(
                lv_ode_all_species,
                (time_span[0], time_span[-1]),
                x0_vals,
                args=(mu_vals, init_alpha),
                t_eval=time_span,
                method='RK45'
            )
            sim_init = sol_init.y  # shape (species, time)

            st.markdown("### Initial Simulation (No interactions)")
            fig_init = go.Figure()
            colors = px.colors.qualitative.Bold
            for i, sp in enumerate(species_cols):
                fig_init.add_trace(go.Scatter(
                    x=time_span,
                    y=observed_data[:, i],
                    mode='markers',
                    name=f"Observed {sp}",
                    marker=dict(symbol="circle", size=6),
                    line=dict(color=colors[i % len(colors)]),
                    opacity=0.7
                ))
                fig_init.add_trace(go.Scatter(
                    x=time_span,
                    y=sim_init[i, :],
                    mode='lines',
                    name=f"Simulated {sp} (no interaction)",
                    line=dict(color=colors[i % len(colors)], dash='dot', width=2),
                    opacity=0.7
                ))
            fig_init.update_layout(
                title="Initial Simulation with Zero Interaction Matrix (α)",
                xaxis_title="Time",
                yaxis_title="Population (OD)",
                template="plotly_white",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_init, use_container_width=True)

            # Flattened initial alpha vector for optimization
            init_alpha_flat = init_alpha.flatten()
            bounds = [(-2.0, 2.0)] * (n_species * n_species)

            run_fit = st.button("Run LV Full Species Fit")

            if run_fit:
                with st.spinner("Fitting Lotka-Volterra full species interaction matrix..."):
                    # Objective function to minimize MSE between observed and simulated
                    def mse_alpha_all_species(alpha_flat):
                        alpha_mat = alpha_flat.reshape((n_species, n_species))
                        sol = solve_ivp(
                            lv_ode_all_species,
                            (time_span[0], time_span[-1]),
                            x0_vals,
                            args=(mu_vals, alpha_mat),
                            t_eval=time_span,
                            method='RK45'
                        )
                        if not sol.success:
                            return 1e10
                        sim_data = sol.y.T  # shape (times, species)
                        mse = np.mean((sim_data - observed_data)**2)
                        return mse

                    res = minimize(mse_alpha_all_species, init_alpha_flat, method='L-BFGS-B', bounds=bounds)

                    if res.success:
                        fitted_alpha = res.x.reshape((n_species, n_species))
                        st.success("Fitting complete!")

                        alpha_df = pd.DataFrame(fitted_alpha, columns=species_cols, index=species_cols)
                        st.markdown("### Fitted interaction matrix (α):")
                        st.dataframe(alpha_df)

                        # Simulate with fitted alpha
                        sol_fit = solve_ivp(
                            lv_ode_all_species,
                            (time_span[0], time_span[-1]),
                            x0_vals,
                            args=(mu_vals, fitted_alpha),
                            t_eval=time_span,
                            method='RK45'
                        )
                        sim_fit = sol_fit.y  # shape (species, times)

                        # Plot observed vs simulated
                        fig_fit = go.Figure()
                        for i, sp in enumerate(species_cols):
                            fig_fit.add_trace(go.Scatter(
                                x=time_span,
                                y=observed_data[:, i],
                                mode='markers',
                                name=f"Observed {sp}",
                                marker=dict(symbol="circle", size=6),
                                line=dict(color=colors[i % len(colors)]),
                                opacity=0.7
                            ))
                            fig_fit.add_trace(go.Scatter(
                                x=time_span,
                                y=sim_fit[i, :],
                                mode='lines',
                                name=f"Simulated {sp} (fitted)",
                                line=dict(color=colors[i % len(colors)], dash='dash', width=2),
                                opacity=0.7
                            ))
                        fig_fit.update_layout(
                            title="Observed vs Simulated (Fitted LV Model)",
                            xaxis_title="Time",
                            yaxis_title="Population (OD)",
                            template="plotly_white",
                            height=600,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_fit, use_container_width=True)

                        # Metrics
                        mse_val = np.mean((sim_fit.T - observed_data) ** 2)
                        # Calculate R^2 per species
                        r_squared_list = []
                        for i in range(n_species):
                            ss_res = np.sum((observed_data[:, i] - sim_fit[i, :])**2)
                            ss_tot = np.sum((observed_data[:, i] - np.mean(observed_data[:, i]))**2)
                            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
                            r_squared_list.append(r2)
                        r2_df = pd.DataFrame({
                            "Species": species_cols,
                            "R²": [f"{r:.4f}" if not np.isnan(r) else "N/A" for r in r_squared_list]
                        })
                        st.markdown(f"**Mean squared error (MSE) across all species:** {mse_val:.5f}")
                        st.markdown("### R² values for each species:")
                        st.dataframe(r2_df, use_container_width=True)

                    else:
                        st.error("Fitting failed. Try different bounds or initial values.")






    else:
        st.warning("Please upload and process data first!")
        st.markdown("Go to the **Data Upload** section to get started.")

# ========== About Tab ==========
with main_tabs[3]:
    st.markdown('<h2 class="section-header">About Cofit Dashboard</h2>', unsafe_allow_html=True)
    st.markdown("""
    ### Description
    The Cofit Dashboard is designed for analyzing microbial growth data from co-culture experiments. 
    It provides tools for:
    - Importing and processing experimental data
    - Visualizing growth curves for individual species and co-cultures
    - Fitting growth models to extract biological parameters
    ### How to Use
    1. **Upload Data**: Start by uploading your replicate data files in CSV or Excel format
    2. **Configure Analysis**: Specify the number of species and processing options
    3. **Analyze Data**: Examine the processed growth curves
    4. **Fit Models**: Extract growth parameters using ODE models
    ### References
    - [Example Publication](https://journals.asm.org/)
    - [Mathematical Models in Microbial Ecology](https://www.sciencedirect.com/)
    """)
    st.info("For questions or support, please contact: example@university.edu")

