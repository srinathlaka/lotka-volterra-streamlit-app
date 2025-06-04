import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings

# Ignore scipy integration and optimization warnings for cleaner UI
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

st.set_page_config(
    page_title="Cofit Dashboard",
    page_icon=":microscope:",
    layout="wide"
)

with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/000000/microscope.png", width=60)
    st.title("Cofit Dashboard")
    section = st.radio("Navigation", ["Data Upload", "Data Visualization", "Cofit"])

st.title("Cofit Dashboard")

def add_pairwise_columns(df_wells):
    cols = df_wells.columns
    pairwise_data = {}
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            col_name = f"{cols[i]}+{cols[j]}"
            pairwise_data[col_name] = df_wells.iloc[:, i] + df_wells.iloc[:, j]
    df_pairs = pd.DataFrame(pairwise_data)
    return pd.concat([df_wells, df_pairs], axis=1)

def fit_growth_rate(time, avg_wells, fit_start, fit_end):
    mask = (time >= fit_start) & (time <= fit_end)
    time_fit = time[mask]
    avg_fit = avg_wells[mask]

    if len(time_fit) < 3:
        return None, "Not enough data points in the fit range."
    
    # Convert to numpy arrays for better numerical stability
    time_fit_np = np.array(time_fit)
    avg_fit_np = np.array(avg_fit)
    
    # Filter out negative or zero values that would cause issues with log transformation
    valid_indices = avg_fit_np > 0
    if sum(valid_indices) < 3:
        return None, "Not enough positive data points for reliable fitting."
    
    time_fit_np = time_fit_np[valid_indices]
    avg_fit_np = avg_fit_np[valid_indices]
    
    # Try both ODE-based fitting and direct exponential fitting
    
    # Method 1: ODE-based approach
    def ode_growth(t, x, mu):
        return mu * x

    def objective(mu_array):
        mu = mu_array[0]
        # Use the first valid data point as initial condition
        z0 = [avg_fit_np[0]]
        try:
            sol = solve_ivp(
                ode_growth,
                [time_fit_np[0], time_fit_np[-1]],
                z0,
                t_eval=time_fit_np,
                args=(mu,),
                method='LSODA',
                rtol=1e-6,
                atol=1e-8
            )
            if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                return 1e6
            model = sol.y[0]
            return np.sum((model - avg_fit_np) ** 2)
        except:
            return 1e6

    # Try different initial guesses for mu
    best_result = None
    best_score = float('inf')
    
    for initial_mu in [0.01, 0.1, 0.5, 1.0]:
        bounds = [(0.0001, 10)]
        res = minimize(objective, [initial_mu], method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_result = res
    
    # Method 2: Direct fit using logarithm (simpler approach as backup)
    try:
        # y = y0*exp(mu*t) => ln(y) = ln(y0) + mu*t
        log_y = np.log(avg_fit_np)
        A = np.vstack([time_fit_np, np.ones(len(time_fit_np))]).T
        mu_direct, log_y0 = np.linalg.lstsq(A, log_y, rcond=None)[0]
        direct_score = np.sum((avg_fit_np - np.exp(log_y0) * np.exp(mu_direct * time_fit_np)) ** 2)
    except:
        direct_score = float('inf')
    
    # Choose best method
    if best_result is not None and best_score <= direct_score:
        return float(best_result.x[0]), None
    elif direct_score < float('inf'):
        return float(mu_direct), None
    else:
        return None, "Fitting failed for both ODE and direct methods."

if section == "Data Upload":
    st.header(":inbox_tray: Data Upload")
    st.markdown("Upload your growth data for each species. Data should include time, well data, and background average.")

    species_count = st.number_input(
        "Select number of species",
        min_value=1,
        max_value=5,
        step=1,
        help="How many different species/strains are in your experiment?"
    )

    if species_count:
        tabs = st.tabs([f"Species {i+1}" for i in range(species_count)])

        for i in range(species_count):
            with tabs[i]:
                st.subheader(f"Species {i+1} Data Upload")
                file = st.file_uploader(
                    f"Upload data for Species {i+1} (No header, first col = time, last col = background average)",
                    type=["csv", "xlsx", "xls"],
                    key=f"upload_{i}",
                    help="Accepted formats: .csv, .xlsx, .xls"
                )

                if file is not None:
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file, header=None)
                    else:
                        df = pd.read_excel(file, header=None)

                    if df.shape[1] < 3:
                        st.error(":warning: File must have at least 3 columns: time, well data(s), and background average.")
                        continue

                    time = df.iloc[:, 0]
                    wells_original = df.iloc[:, 1:-1]
                    background_avg = df.iloc[:, -1]

                    wells_original.columns = [f"x{idx+1}" for idx in range(wells_original.shape[1])]
                    wells_augmented = add_pairwise_columns(wells_original)

                    df_augmented = pd.concat([time, wells_augmented, background_avg], axis=1)
                    cols = ["time"] + list(wells_augmented.columns) + ["background_avg"]
                    df_augmented.columns = cols

                    st.markdown("**Preview of data with original and pairwise summed wells (before background subtraction):**")
                    st.dataframe(df_augmented.head(), use_container_width=True)

                    with st.expander("Background Subtraction & Data Preview", expanded=True):
                        subtract_bg = st.toggle(
                            "Subtract background average from well data?",
                            value=False,
                            key=f"bg_subtract_{i}",
                            help="Subtracts the background average column from each well's data."
                        )
                        if subtract_bg:
                            wells_corrected = wells_augmented.subtract(background_avg, axis=0)
                            st.info("Background-subtracted well data preview:")
                            st.dataframe(wells_corrected.head(), use_container_width=True)
                        else:
                            wells_corrected = wells_augmented
                            st.info("Original well data preview:")
                            st.dataframe(wells_corrected.head(), use_container_width=True)

                    avg_wells = wells_corrected.mean(axis=1)
                    sd_wells = wells_corrected.std(axis=1)

                    st.markdown("---")
                    st.subheader(":bar_chart: Well Data Visualization")
                    fig = go.Figure()
                    color_palette = px.colors.qualitative.Plotly
                    for idx, col in enumerate(wells_corrected.columns):
                        fig.add_trace(go.Scatter(
                            x=time,
                            y=wells_corrected[col],
                            mode='lines',
                            line=dict(width=1, color=color_palette[idx % len(color_palette)]),
                            name=col,
                            showlegend=True
                        ))
                    fig.update_layout(
                        title=f"Individual Well Curves ({'Background Subtracted' if subtract_bg else 'Original'}) - Species {i+1}",
                        xaxis_title="Time",
                        yaxis_title="Optical Density (OD)",
                        legend_title="Wells",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fig_avg_sd = go.Figure()
                    fig_avg_sd.add_trace(go.Scatter(
                        x=time,
                        y=avg_wells,
                        mode='lines',
                        line=dict(width=3, color='blue'),
                        name='Average wells',
                        fill='tozeroy',
                        fillcolor='rgba(0,0,255,0.1)'
                    ))
                    fig_avg_sd.add_trace(go.Scatter(
                        x=time,
                        y=sd_wells,
                        mode='lines',
                        line=dict(width=3, color='red', dash='dash'),
                        name='Standard Deviation'
                    ))
                    fig_avg_sd.update_layout(
                        title=f"Average and Standard Deviation of Wells - Species {i+1}",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_avg_sd, use_container_width=True)

                    st.markdown("---")
                    st.subheader(":chart_with_upwards_trend: Exponential Growth Fitting")
                    
                    # Use min/max time values
                    t_min = float(time.min())
                    t_max = float(time.max())
                    
                    # Use sliders for more intuitive range selection
                    st.markdown("#### Select fitting range:")
                    st.markdown("Use the slider below to select the exponential growth phase of your data")
                    
                    # Create a preview plot for range selection
                    fig_preview = go.Figure()
                    fig_preview.add_trace(go.Scatter(
                        x=time, 
                        y=avg_wells,
                        mode='lines+markers',
                        name='Data',
                        marker=dict(size=3),
                        line=dict(width=1)
                    ))
                    fig_preview.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_preview, use_container_width=True)
                    
                    # Create range slider
                    fit_range = st.slider(
                        "Fit Range",
                        min_value=t_min,
                        max_value=t_max,
                        value=(t_min, t_max),
                        step=(t_max-t_min)/100,
                        format="%.2f",
                        key=f"fit_range_slider_{i}"
                    )
                    fit_start, fit_end = fit_range
                    
                    # Add advanced options in expander
                    with st.expander("Advanced Fitting Options"):
                        col1, col2 = st.columns(2)
                        with col1:
                            fit_start = st.number_input(
                                f"Fit start time for Species {i+1}",
                                min_value=t_min,
                                max_value=t_max,
                                value=fit_start,
                                step=0.1,
                                key=f"fit_start_{i}",
                                help="Start time for fitting."
                            )
                        with col2:
                            fit_end = st.number_input(
                                f"Fit end time for Species {i+1}",
                                min_value=t_min,
                                max_value=t_max,
                                value=fit_end,
                                step=0.1,
                                key=f"fit_end_{i}",
                                help="End time for fitting."
                            )
                        
                        show_extrapolation = st.checkbox(
                            "Show extrapolated curve over full time range", 
                            value=False,
                            key=f"show_extrapolation_{i}",
                            help="If checked, the fit curve will be shown for the entire time range, not just the fit range."
                        )
                        use_log_scale = st.checkbox(
                            "Use log scale for visualization", 
                            value=False,
                            key=f"log_scale_{i}",
                            help="Log scale can help visualize exponential growth as a straight line"
                        )
                        
                        st.markdown("""
                        <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px">
                            <b>💡 Tips for better fitting:</b>
                            <ul>
                                <li>Select only the exponential growth phase (before plateau)</li>
                                <li>Try using log scale to identify the linear region</li>
                                <li>Exclude lag phase from fit range</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    if fit_end <= fit_start:
                        st.error(":x: Fit end time must be greater than fit start time.")                    
                    if st.button(f"Run exponential growth fit for Species {i+1}", key=f"fit_button_{i}"):
                        mu, msg = fit_growth_rate(time, avg_wells, fit_start, fit_end)
                        if mu is not None:
                            st.success(f"\U0001F389 Fitting successful! Estimated mu: {mu:.5f}")
                            st.session_state.setdefault(f"species_{i}_data", {})
                            st.session_state[f"species_{i}_data"]["fit_mu"] = mu
                            st.session_state[f"species_{i}_data"]["fit_msg"] = None
                            st.session_state[f"species_{i}_data"]["fit_start"] = fit_start
                            st.session_state[f"species_{i}_data"]["fit_end"] = fit_end
                            # Store extrapolation preference
                            show_extrapolation = st.session_state.get(f"show_extrapolation_{i}", False)
                            st.session_state[f"species_{i}_data"]["show_extrapolation"] = show_extrapolation                        
                        else:
                            st.error(f"Fitting failed: {msg}")
                            st.session_state.setdefault(f"species_{i}_data", {})
                            st.session_state[f"species_{i}_data"]["fit_mu"] = None
                            st.session_state[f"species_{i}_data"]["fit_msg"] = msg
                            
                    fit_mu = st.session_state.get(f"species_{i}_data", {}).get("fit_mu", None)
                    if fit_mu is not None:
                        # Get fit range from session state
                        fit_start_plot = st.session_state[f"species_{i}_data"].get("fit_start", float(time.min()))
                        fit_end_plot = st.session_state[f"species_{i}_data"].get("fit_end", float(time.max()))
                        
                        # Find data at fit start time
                        start_idx = (time >= fit_start_plot).idxmin() if fit_start_plot > time.min() else 0
                        initial_value = avg_wells.iloc[start_idx]                        # Check if we should show extrapolation
                        show_extrapolation = st.session_state[f"species_{i}_data"].get("show_extrapolation", False)
                        
                        # Set the start and end points for plotting
                        plot_start = float(time.min()) if show_extrapolation else fit_start_plot
                        plot_end = float(time.max()) if show_extrapolation else fit_end_plot
                        
                        # Create a finer time range for plotting
                        t_plot_fit = np.linspace(plot_start, plot_end, 500)
                        
                        # Generate the solution
                        z0 = [max(initial_value, 1e-3)]
                        sol = solve_ivp(
                            lambda t, x: fit_mu * x, 
                            [plot_start, plot_end], 
                            z0, 
                            t_eval=t_plot_fit,
                            method='RK45',
                            rtol=1e-6,
                            atol=1e-8
                        )
                        
                        # Create figure with both data and fit
                        fig_fit = go.Figure()
                        
                        # Plot all data points
                        fig_fit.add_trace(go.Scatter(
                            x=time,
                            y=avg_wells,
                            mode='markers',
                            name='All data',
                            marker=dict(size=5, color='lightgray', symbol='circle'),
                            opacity=0.5
                        ))
                        
                        # Highlight fit data range
                        mask = (time >= fit_start_plot) & (time <= fit_end_plot)
                        fig_fit.add_trace(go.Scatter(
                            x=time[mask],
                            y=avg_wells[mask],
                            mode='markers',
                            name='Fit data range',
                            marker=dict(size=6, color='black', symbol='circle')
                        ))
                          # Plot the fit curve
                        fig_fit.add_trace(go.Scatter(
                            x=t_plot_fit,
                            y=sol.y[0],
                            mode='lines',
                            name=f'Fit: μ={fit_mu:.5f}',
                            line=dict(width=3, color='green')
                        ))                        # Apply log scale if selected
                        use_log_scale = st.session_state.get(f"log_scale_{i}", False)
                        
                        fig_fit.update_layout(
                            title=f"Exponential Growth Fit - Species {i+1}",
                            xaxis_title="Time",
                            yaxis_title="Optical Density / Population",
                            template="plotly_white",
                            yaxis_type="log" if use_log_scale else "linear"
                        )
                        st.plotly_chart(fig_fit, use_container_width=True)

                        # Add visualization of the fit quality                        
                        # # Calculate predicted values only within the fit range
                        mask_fit = (time >= fit_start_plot) & (time <= fit_end_plot)
                        actual_times_fit = np.array(time[mask_fit])
                        z0_actual = [max(initial_value, 1e-3)]
                        sol_actual = solve_ivp(
                            lambda t, x: fit_mu * x, 
                            [fit_start_plot, fit_end_plot], 
                            z0_actual, 
                            t_eval=actual_times_fit,
                            method='RK45',
                            rtol=1e-6,
                            atol=1e-8
                        )
                          # Calculate residuals and R^2 only for the fit range
                        predicted = sol_actual.y[0]
                        actual = np.array(avg_wells[mask_fit])
                        residuals = actual - predicted
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((actual - np.mean(actual))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        rmse = np.sqrt(np.mean(residuals**2))
                        
                        # Update plot title with metrics
                        fig_fit.update_layout(
                            title=f"Exponential Growth Fit - Species {i+1} (μ={fit_mu:.5f}, R²={r_squared:.3f}, RMSE={rmse:.3f})",
                            xaxis_title="Time",
                            yaxis_title="Optical Density / Population",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_fit, use_container_width=True)
                          # Display residuals plot - only for data in the fit range
                        fig_residuals = go.Figure()
                        fig_residuals.add_trace(go.Scatter(
                            x=time[mask_fit],
                            y=residuals,
                            mode='markers',
                            marker=dict(size=5, color='blue'),
                            name='Residuals'
                        ))
                        fig_residuals.add_trace(go.Scatter(
                            x=[fit_start_plot, fit_end_plot],
                            y=[0, 0],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Zero line'
                        ))
                        fig_residuals.update_layout(
                            title="Residuals (Actual - Predicted)",
                            xaxis_title="Time",
                            yaxis_title="Residual Value",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_residuals, use_container_width=True)
                        
                        # Add a message about fit quality
                        if r_squared > 0.95:
                            st.success(f"Excellent fit quality (R² = {r_squared:.3f})")
                        elif r_squared > 0.85:
                            st.info(f"Good fit quality (R² = {r_squared:.3f})")
                        elif r_squared > 0.7:
                            st.warning(f"Moderate fit quality (R² = {r_squared:.3f}). Consider adjusting the fit range.")
                        else:
                            st.error(f"Poor fit quality (R² = {r_squared:.3f}). The exponential model may not be appropriate for this data, or you need to adjust the fit range.")
                        
                        # Create an expander with fit details
                        with st.expander("Fit Details"):
                            st.markdown(f"""
                            - **Growth rate (μ):** {fit_mu:.5f}
                            - **R²:** {r_squared:.5f}
                            - **RMSE:** {rmse:.5f}
                            - **Doubling time:** {np.log(2)/fit_mu:.5f} time units
                            - **Fit range:** {fit_start_plot:.2f} to {fit_end_plot:.2f}
                            """)
                            
                            # Display a more technical explanation
                            st.markdown("""
                            **About the fit:**
                            
                            The exponential growth model follows: dN/dt = μN, where:
                            - N is the population or optical density
                            - μ is the growth rate constant
                            - The solution is N(t) = N₀*exp(μ*t)
                            
                            For bacterial growth, the doubling time is calculated as ln(2)/μ
                            """)
