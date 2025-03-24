import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import traceback

# Set page configuration
st.set_page_config(page_title="ES-Agent: Minimal Mode", 
                   page_icon="ud83dudcca", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

# Title and description
st.title("ES-Agent: Earned Schedule Prediction Analysis")
st.markdown("""
**Earned Schedule (ES)** is an advanced project control technique that extends traditional 
Earned Value Management (EVM) by measuring schedule performance in time units.

This is a simplified version of the ES-Agent application.
""")

# Warning about limited functionality
st.warning("This is a minimal version of the ES-Agent with limited functionality. Some advanced features may not be available.")

# Initialize session state variables if they don't exist
if 'has_data' not in st.session_state:
    st.session_state.has_data = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'project_name' not in st.session_state:
    st.session_state.project_name = "New Project"

# Create example project data
def generate_example_project(project_type="on_track", periods=12):
    """Generate example project data with different performance patterns"""
    time_periods = list(range(1, periods + 1))
    planned_duration = periods
    
    # Base S-curve for PV (planned value)
    pv_values = []
    for t in time_periods:
        # Create an S-curve for planned value
        progress = t / planned_duration
        if progress < 0.2:
            # Slow start
            pv_factor = progress * 1.5
        elif progress > 0.8:
            # Slow finish
            pv_factor = 0.7 + 0.3 * (progress - 0.8) / 0.2
        else:
            # Steady middle progress
            pv_factor = 0.3 + 0.4 * (progress - 0.2) / 0.6
        pv_values.append(100 * pv_factor)
    
    # Generate EV (earned value) based on project type
    ev_values = []
    if project_type == "on_track":
        # Project progressing as planned
        ev_values = [pv * 0.98 for pv in pv_values]  # Slightly behind but essentially on track
    elif project_type == "ahead":
        # Project ahead of schedule
        ev_values = [min(pv * 1.2, 100) for pv in pv_values]  # 20% ahead of schedule
    elif project_type == "behind":
        # Project behind schedule
        ev_values = [pv * 0.7 for pv in pv_values]  # 30% behind schedule
    elif project_type == "recovery":
        # Project starts behind but recovers
        ev_values = []
        for i, pv in enumerate(pv_values):
            progress = (i + 1) / periods
            if progress < 0.4:
                # Start 40% behind
                factor = 0.6
            else:
                # Gradually recover to 95% by the end
                factor = 0.6 + 0.35 * (progress - 0.4) / 0.6
            ev_values.append(pv * factor)
    elif project_type == "slipping":
        # Project starts well but then slips
        ev_values = []
        for i, pv in enumerate(pv_values):
            progress = (i + 1) / periods
            if progress < 0.3:
                # Start on track
                factor = 1.0
            else:
                # Gradually slip to 70% by the end
                factor = 1.0 - 0.3 * (progress - 0.3) / 0.7
            ev_values.append(pv * factor)
    
    # Calculate ES (earned schedule) for each time period
    es_values = []
    for i, ev in enumerate(ev_values):
        # Find where this EV would occur on the PV curve
        for j in range(len(pv_values)):
            if j == len(pv_values) - 1:
                es = j + 1
                break
            if pv_values[j] <= ev and pv_values[j+1] > ev:
                # Linear interpolation
                es = j + 1 + (ev - pv_values[j]) / (pv_values[j+1] - pv_values[j])
                break
            if pv_values[j] > ev:
                es = j * (ev / pv_values[j])
                break
        es_values.append(min(es, periods))
    
    # Calculate SPI(t) values
    spi_t_values = [es / t if t > 0 else 1.0 for t, es in zip(time_periods, es_values)]
    
    # Create dataframe
    df = pd.DataFrame({
        'Time': time_periods,
        'PV': pv_values,
        'EV': ev_values,
        'ES': es_values,
        'SPI(t)': spi_t_values,
        'PD': [planned_duration] * len(time_periods),
        'AT': [periods] * len(time_periods),
    })
    
    return df

# Example project descriptions for the dropdown
example_projects = {
    "Select an example project": None,
    "On Track Project": {"type": "on_track", "description": "A project that is progressing close to plan."},
    "Ahead of Schedule": {"type": "ahead", "description": "A project that is performing better than planned."},
    "Behind Schedule": {"type": "behind", "description": "A project that is significantly behind schedule."},
    "Recovery Project": {"type": "recovery", "description": "A project that started poorly but is recovering."},
    "Slipping Project": {"type": "slipping", "description": "A project that started well but is falling behind."},
}

# Create sidebar for data input
with st.sidebar:
    st.header("Data Input")
    
    # Example project selector
    st.subheader("Example Projects")
    example_selection = st.selectbox(
        "Select an example project",
        list(example_projects.keys())
    )
    
    if example_selection != "Select an example project" and example_projects[example_selection] is not None:
        project_info = example_projects[example_selection]
        st.info(project_info["description"])
        
        if st.button("Load Example Project"):
            # Generate the example data
            st.session_state.data = generate_example_project(project_info["type"])
            st.session_state.has_data = True
            st.session_state.project_name = example_selection
            st.success(f"Loaded example: {example_selection}")
    
    st.markdown("---")
    
    # Project name input
    project_name = st.text_input("Project Name", value=st.session_state.project_name)
    st.session_state.project_name = project_name
    
    # File upload
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload Excel file with project data", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Try to load the file
            df = pd.read_excel(uploaded_file)
            
            # Check if required columns exist
            required_columns = ['Time', 'PV', 'EV']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success("File loaded successfully!")
                st.session_state.data = df
                st.session_state.has_data = True
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Manual inputs
    st.subheader("Manual Inputs")
    st.markdown("For simple calculations and testing:")
    
    col1, col2 = st.columns(2)
    with col1:
        at = st.number_input("AT (Actual Time)", min_value=0.0, value=10.0, step=0.1)
        pd_value = st.number_input("PD (Planned Duration)", min_value=0.0, value=20.0, step=0.1)
    with col2:
        es = st.number_input("ES (Earned Schedule)", min_value=0.0, value=8.0, step=0.1)
    
    if st.button("Use Manual Inputs"):
        # Create a simple dataset with manual inputs
        time_periods = list(range(1, int(at) + 1))
        pv_values = [pd_value * (t / pd_value) ** 0.5 for t in time_periods]  # Simple S-curve for PV
        ev_values = [pv_values[int(min(es, len(pv_values)-1))] * (t / at) for t in time_periods]  # Simple EV curve
        
        data = {
            'Time': time_periods,
            'PV': pv_values,
            'EV': ev_values,
            'AT': at,
            'ES': es,
            'PD': pd_value,
            'SPI(t)': es / at if at > 0 else 1.0
        }
        st.session_state.data = pd.DataFrame(data)
        st.session_state.has_data = True
        st.success("Manual data generated!")

# Main content area
if st.session_state.has_data and st.session_state.data is not None:
    data = st.session_state.data
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["ES Dashboard", "Advanced Charts", "Data View"])
    
    with tab1:
        st.header("Earned Schedule Metrics Dashboard")
        
        # Calculate basic ES metrics
        try:
            # Get the latest period
            latest_data = data.iloc[-1]
            
            # Extract metrics or use default values
            at = latest_data.get('AT', latest_data.get('Time', 0))
            es = latest_data.get('ES', 0)
            pd_value = latest_data.get('PD', 0)
            
            # Calculate basic metrics
            spi_t = es / at if at > 0 else 1.0
            completion_pct = (es / pd_value) * 100 if pd_value > 0 else 0
            
            # Simple forecasts
            ieac_current = pd_value / spi_t if spi_t > 0 else "N/A"
            ieac_pf1 = at + (pd_value - es) if pd_value > es else at
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Earned Schedule (ES)", f"{es:.2f}")
                st.metric("Actual Time (AT)", f"{at:.2f}")
            
            with col2:
                st.metric("Schedule Performance Index SPI(t)", f"{spi_t:.2f}", 
                         delta=f"{spi_t - 1:.2f}", 
                         delta_color="normal" if spi_t >= 1 else "inverse")
                st.metric("Completion %", f"{completion_pct:.1f}%")
            
            with col3:
                st.metric("Planned Duration (PD)", f"{pd_value:.2f}")
                status = "Ahead" if es > at else "Behind" if es < at else "On Track"
                st.metric("Schedule Status", status, 
                         delta=f"{es - at:.2f}", 
                         delta_color="normal" if es >= at else "inverse")
            
            with col4:
                st.metric("Est. Completion (Current)", f"{ieac_current:.2f}" if isinstance(ieac_current, (int, float)) else ieac_current, 
                         delta=f"{float(ieac_current) - pd_value:.2f}" if isinstance(ieac_current, (int, float)) else None,
                         delta_color="inverse" if isinstance(ieac_current, (int, float)) and float(ieac_current) > pd_value else "normal")
                st.metric("Est. Completion (PF=1)", f"{ieac_pf1:.2f}" if isinstance(ieac_pf1, (int, float)) else ieac_pf1,
                         delta=f"{float(ieac_pf1) - pd_value:.2f}" if isinstance(ieac_pf1, (int, float)) else None,
                         delta_color="inverse" if isinstance(ieac_pf1, (int, float)) and float(ieac_pf1) > pd_value else "normal")
            
            # Interpretation
            st.subheader("Performance Interpretation")
            if spi_t > 1.05:
                st.success(f"The project is **ahead of schedule** with SPI(t) = {spi_t:.2f}. It's accomplishing {(spi_t-1)*100:.1f}% more than planned each period.")
            elif spi_t >= 0.95:
                st.info(f"The project is **on schedule** with SPI(t) = {spi_t:.2f}. It's maintaining pace with the plan.")
            elif spi_t >= 0.8:
                st.warning(f"The project is **behind schedule** with SPI(t) = {spi_t:.2f}. It's accomplishing {(1-spi_t)*100:.1f}% less than planned each period.")
            else:
                st.error(f"The project is **significantly behind schedule** with SPI(t) = {spi_t:.2f}. It's accomplishing {(1-spi_t)*100:.1f}% less than planned each period.")
            
            # S-Curve Visualization
            st.subheader("S-Curve Analysis")
            
            # Create figure
            fig = px.line(data, x='Time', y=['PV', 'EV'], 
                           title='Project S-Curve: Planned vs Actual Progress',
                           labels={'value': 'Value', 'variable': 'Metric'},
                           line_shape='linear')
            
            # Improve styling
            fig.update_layout(
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display ES and SPI(t) charts side by side
            col1, col2 = st.columns(2)
            
            with col1:
                # ES vs AT Chart
                if 'ES' in data.columns:
                    fig_es = px.scatter(data, x='Time', y='ES', title='Earned Schedule vs Actual Time')
                    
                    # Add reference line (ES = AT, on schedule)
                    fig_es.add_trace(go.Scatter(
                        x=[data['Time'].min(), data['Time'].max()],
                        y=[data['Time'].min(), data['Time'].max()],
                        mode='lines',
                        name='On Schedule (ES = AT)',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                    
                    # Improve styling
                    fig_es.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig_es, use_container_width=True)
            
            with col2:
                # SPI(t) trend chart
                if 'SPI(t)' in data.columns:
                    fig_spi = px.line(data, x='Time', y='SPI(t)', title='Schedule Performance Index Trend')
                    
                    # Add reference line (SPI(t) = 1, on schedule)
                    fig_spi.add_trace(go.Scatter(
                        x=[data['Time'].min(), data['Time'].max()],
                        y=[1, 1],
                        mode='lines',
                        name='On Schedule (SPI(t) = 1)',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                    
                    # Improve styling
                    fig_spi.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig_spi, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            st.code(traceback.format_exc())
    
    with tab2:
        st.header("Advanced Visualizations and Analysis")
        
        # Choose which advanced charts to display
        chart_options = st.multiselect(
            "Select charts to display",
            ["ES vs. Time 3D", "Performance Heatmap", "Forecast Simulation", "ES to PD Progress", "Schedule Variance"],
            default=["ES to PD Progress", "Schedule Variance"]
        )
        
        if "ES to PD Progress" in chart_options:
            # Visualization of ES progress to PD
            st.subheader("Schedule Progress Towards Completion")
            
            latest_es = data['ES'].iloc[-1] if 'ES' in data.columns else 0
            pd_value = data['PD'].iloc[-1] if 'PD' in data.columns else 1
            progress_pct = (latest_es / pd_value) * 100 if pd_value > 0 else 0
            
            # Create a gauge chart for progress
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=progress_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Schedule Progress"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "#ffcccb"},
                        {'range': [33, 67], 'color': "#ffffcc"},
                        {'range': [67, 100], 'color': "#d4f1c5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100*(data['Time'].iloc[-1]/pd_value) if pd_value > 0 else 0
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            if progress_pct < 90 * (data['Time'].iloc[-1]/pd_value) * 100:
                st.warning(f"The project is progressing at {progress_pct:.1f}% of its planned duration, but time elapsed is {(data['Time'].iloc[-1]/pd_value) * 100:.1f}% of the schedule.")
            else:
                st.success(f"The project is progressing well at {progress_pct:.1f}% of its planned duration.")
        
        if "Schedule Variance" in chart_options:
            # Visualization of schedule variance over time
            st.subheader("Schedule Variance Over Time")
            
            if 'ES' in data.columns and 'Time' in data.columns:
                # Calculate SV(t) - Schedule Variance in time units
                sv_t = data['ES'] - data['Time']
                
                # Create a dataframe with the variance
                sv_df = pd.DataFrame({
                    'Time': data['Time'],
                    'SV(t)': sv_t
                })
                
                # Plot the variance
                fig = px.bar(
                    sv_df, 
                    x='Time', 
                    y='SV(t)', 
                    title='Schedule Variance (time) Over Time',
                    labels={'SV(t)': 'Schedule Variance (time units)'},
                    color='SV(t)',
                    color_continuous_scale=['red', 'yellow', 'green'],
                )
                
                # Add a reference line at SV(t) = 0 (on schedule)
                fig.add_trace(go.Scatter(
                    x=[data['Time'].min(), data['Time'].max()],
                    y=[0, 0],
                    mode='lines',
                    name='On Schedule',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                latest_sv = sv_t.iloc[-1]
                if latest_sv > 0.5:
                    st.success(f"The project is currently **ahead of schedule** by {latest_sv:.2f} time units.")
                elif latest_sv >= -0.5:
                    st.info(f"The project is currently **on schedule** with a variance of {latest_sv:.2f} time units.")
                else:
                    st.warning(f"The project is currently **behind schedule** by {abs(latest_sv):.2f} time units.")
        
        if "Performance Heatmap" in chart_options:
            # Create a performance heatmap
            st.subheader("Performance Heatmap")
            
            # Prepare data for the heatmap
            if 'SPI(t)' in data.columns:
                performance_data = []
                for i, row in data.iterrows():
                    performance_data.append({
                        'Time': row['Time'],
                        'Metric': 'SPI(t)',
                        'Value': row['SPI(t)'],
                    })
                
                if 'ES' in data.columns and 'Time' in data.columns:
                    for i, row in data.iterrows():
                        performance_data.append({
                            'Time': row['Time'],
                            'Metric': 'ES/AT',
                            'Value': row['ES'] / row['Time'] if row['Time'] > 0 else 1.0,
                        })
                
                # Create heatmap dataframe
                heatmap_df = pd.DataFrame(performance_data)
                
                # Plot heatmap
                fig = px.density_heatmap(
                    heatmap_df,
                    x='Time',
                    y='Metric',
                    z='Value',
                    title='Performance Metrics Heatmap',
                    color_continuous_scale=['red', 'yellow', 'green'],
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        if "Forecast Simulation" in chart_options:
            # Monte Carlo simulation for forecast
            st.subheader("Forecast Simulation")
            
            # Simulation parameters
            st.write("Simple Monte Carlo simulation for completion date")
            num_simulations = st.slider("Number of simulations", 100, 1000, 200, 100)
            
            # Perform a basic Monte Carlo simulation
            try:
                if 'SPI(t)' in data.columns and 'PD' in data.columns and 'ES' in data.columns:
                    # Get the latest metrics
                    latest_spi_t = data['SPI(t)'].iloc[-1]
                    latest_es = data['ES'].iloc[-1]
                    pd_value = data['PD'].iloc[-1]
                    remaining_work = pd_value - latest_es
                    
                    # Basic simulation model
                    np.random.seed(42)  # For reproducibility
                    
                    # Base the simulation on historical SPI(t) variability
                    historical_spi_t = data['SPI(t)'].dropna().tolist()
                    
                    if len(historical_spi_t) >= 3:  # Only if we have enough data points
                        spi_t_std = np.std(historical_spi_t) * 1.5  # Amplify variability a bit
                        
                        # Run simulations
                        simulated_completions = []
                        for _ in range(num_simulations):
                            # Simulate future SPI(t) based on current value and historical variability
                            future_spi_t = np.random.normal(latest_spi_t, spi_t_std)
                            future_spi_t = max(0.1, future_spi_t)  # Ensure SPI(t) is not too small
                            
                            # Calculate completion based on simulated SPI(t)
                            simulated_completion = data['Time'].iloc[-1] + (remaining_work / future_spi_t)
                            simulated_completions.append(simulated_completion)
                        
                        # Create chart of simulation results
                        fig = px.histogram(
                            x=simulated_completions,
                            title='Simulation of Possible Completion Times',
                            labels={'x': 'Completion Time', 'y': 'Frequency'},
                            opacity=0.8,
                        )
                        
                        # Add reference line for planned duration
                        fig.add_vline(
                            x=pd_value,
                            line_dash='dash',
                            line_color='red',
                            annotation_text='Planned Duration',
                            annotation_position='top',
                        )
                        
                        # Add percentile lines
                        p10 = np.percentile(simulated_completions, 10)
                        p50 = np.percentile(simulated_completions, 50)
                        p90 = np.percentile(simulated_completions, 90)
                        
                        fig.add_vline(x=p10, line_dash='dot', line_color='green',
                                      annotation_text='P10: ' + f"{p10:.2f}", annotation_position='bottom')
                        fig.add_vline(x=p50, line_dash='dot', line_color='blue',
                                      annotation_text='P50: ' + f"{p50:.2f}", annotation_position='bottom')
                        fig.add_vline(x=p90, line_dash='dot', line_color='orange',
                                      annotation_text='P90: ' + f"{p90:.2f}", annotation_position='bottom')
                        
                        fig.update_layout(
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display analysis results
                        st.markdown(f"""
                        ### Simulation Results:
                        - There is a **10%** chance of completing by **{p10:.2f}** time units (optimistic)
                        - There is a **50%** chance of completing by **{p50:.2f}** time units (most likely)
                        - There is a **90%** chance of completing by **{p90:.2f}** time units (pessimistic)
                        - Planned duration: **{pd_value:.2f}** time units
                        """)
                        
                        # Probability of meeting planned duration
                        prob_meeting_pd = sum(1 for x in simulated_completions if x <= pd_value) / len(simulated_completions) * 100
                        
                        if prob_meeting_pd > 75:
                            st.success(f"Probability of meeting planned duration: {prob_meeting_pd:.1f}%")
                        elif prob_meeting_pd > 25:
                            st.warning(f"Probability of meeting planned duration: {prob_meeting_pd:.1f}%")
                        else:
                            st.error(f"Probability of meeting planned duration: {prob_meeting_pd:.1f}%")
                    else:
                        st.info("Not enough historical SPI(t) data for simulation. Need at least 3 data points.")
                else:
                    st.info("Simulation requires SPI(t), ES and PD data to be available.")
            except Exception as e:
                st.error(f"Error in simulation: {str(e)}")
        
        if "ES vs. Time 3D" in chart_options:
            # 3D visualization of ES vs. Time
            st.subheader("3D Visualization of Project Progress")
            
            if 'ES' in data.columns and 'Time' in data.columns and 'SPI(t)' in data.columns:
                try:
                    # Create a 3D scatter plot
                    fig = go.Figure(data=[go.Scatter3d(
                        x=data['Time'],
                        y=data['ES'],
                        z=data['SPI(t)'],
                        mode='markers+lines',
                        marker=dict(
                            size=5,
                            color=data['SPI(t)'],
                            colorscale='Viridis',
                            opacity=0.8,
                            showscale=True,
                            colorbar=dict(title="SPI(t)")
                        )
                    )])
                    
                    # Update layout
                    fig.update_layout(
                        title='3D Project Progress',
                        scene=dict(
                            xaxis_title='Actual Time (AT)',
                            yaxis_title='Earned Schedule (ES)',
                            zaxis_title='Schedule Performance Index (SPI(t))',
                            xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
                            yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
                            zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)')
                        ),
                        height=600,
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating 3D visualization: {str(e)}")
            else:
                st.info("3D visualization requires ES, Time, and SPI(t) data.")
    
    with tab3:
        st.header("Project Data")
        st.dataframe(data)
        
        # Allow download of data as CSV
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"{st.session_state.project_name.replace(' ', '_')}_data.csv",
            mime='text/csv',
        )
        
        # Data analysis tools
        st.subheader("Data Analysis")
        
        # Display statistics
        if st.checkbox("Show Statistics"):
            st.write("Summary Statistics:")
            st.write(data.describe())
        
        # Correlation analysis
        if st.checkbox("Show Correlation Analysis"):
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = data[numeric_cols].corr()
                
                # Plot heatmap
                fig = px.imshow(corr_matrix, 
                               text_auto=True, 
                               title="Correlation Matrix",
                               color_continuous_scale='RdBu_r',
                               labels=dict(x="Variables", y="Variables", color="Correlation"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Highlight important correlations
                st.write("Strong Correlations (absolute value > 0.7):")
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corrs.append({
                                'Variables': f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}",
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if strong_corrs:
                    st.table(pd.DataFrame(strong_corrs))
                else:
                    st.write("No strong correlations found.")
            else:
                st.write("Not enough numeric columns for correlation analysis.")

else:
    # No data yet, show instructions
    st.info("Please upload project data using the sidebar or enter manual inputs to get started.")
    
    # Show instructions
    st.markdown("""
    ### Getting Started with ES-Agent:
    
    1. **Upload your project data** using the file uploader in the sidebar
       - Prepare an Excel file with Time, PV, and EV columns
       - Optionally include AC for cost analysis
       
    2. **Or enter data manually** using the form in the sidebar
       - AT (Actual Time): Current time period
       - ES (Earned Schedule): Current earned schedule value
       - PD (Planned Duration): Total planned duration
       
    3. **View the analysis** once data is loaded
       - ES Dashboard: Key performance indicators
       - Data View: Raw data table
    """)
    
    # Display a sample visualization
    st.subheader("Sample Visualization (for reference)")
    
    # Generate sample data
    sample_time = list(range(1, 11))
    sample_pv = [10*t**0.5 for t in sample_time]
    sample_ev = [8*t**0.5 for t in sample_time]
    
    # Create a simple dataframe
    sample_df = pd.DataFrame({
        'Time': sample_time,
        'PV': sample_pv,
        'EV': sample_ev
    })
    
    # Plot sample data
    fig = px.line(sample_df, x='Time', y=['PV', 'EV'], 
                title='Sample S-Curve Chart',
                labels={'value': 'Value', 'variable': 'Metric'},
                line_shape='linear')
    st.plotly_chart(fig, use_container_width=True)
