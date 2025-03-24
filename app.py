import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import io
import sqlite3

# Import utilities
from utils.es_calculations import EarnedScheduleCalculator
from utils.prediction_models import ESPredictionModels
from utils.data_processing import ESDataProcessor

# Set up database connection
def get_db_connection():
    conn = sqlite3.connect('es_agent.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS project_data (
        id INTEGER PRIMARY KEY,
        project_id INTEGER,
        period_date TIMESTAMP,
        time_period REAL,
        pv REAL,
        ev REAL,
        ac REAL,
        es REAL,
        spi_t REAL,
        pd REAL,
        at REAL,
        ed REAL,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    );
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Set page configuration
st.set_page_config(page_title="ES-Agent: Earned Schedule Prediction Analysis", 
                   page_icon="📊", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

# Title and description
st.title("ES-Agent: Earned Schedule Prediction Analysis")
st.markdown("""
**Earned Schedule (ES)** is an advanced project control technique that extends traditional 
Earned Value Management (EVM) by measuring schedule performance in time units.

This AI Agent helps project managers predict schedule outcomes by leveraging ES metrics 
along with multiple forecasting models to improve decision-making.
""")

# Check if pmdarima is available and warn the user if it's not
try:
    from utils.prediction_models import PMDARIMA_AVAILABLE
    if not PMDARIMA_AVAILABLE:
        st.warning("Some advanced forecasting features may be limited because the pmdarima package is not installed. The application will still work with standard forecasting models.")
except ImportError:
    st.warning("Some advanced forecasting features may be limited. The application will still work with standard forecasting models.")

# Initialize session state variables if they don't exist
if 'data_processor' not in st.session_state:
    try:
        st.session_state.data_processor = ESDataProcessor()
    except Exception as e:
        st.error(f"Error initializing data processor: {str(e)}")
        # Create fallback
        st.session_state.data_processor = None
    
if 'es_calculator' not in st.session_state:
    try:
        st.session_state.es_calculator = EarnedScheduleCalculator()
    except Exception as e:
        st.error(f"Error initializing ES calculator: {str(e)}")
        # Create fallback
        st.session_state.es_calculator = None
    
if 'prediction_models' not in st.session_state:
    try:
        st.session_state.prediction_models = ESPredictionModels()
    except Exception as e:
        st.error(f"Error initializing prediction models: {str(e)}")
        # Create fallback
        st.session_state.prediction_models = None
    
if 'has_data' not in st.session_state:
    st.session_state.has_data = False
    
if 'latest_metrics' not in st.session_state:
    st.session_state.latest_metrics = {}
    
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
    
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

if 'project_name' not in st.session_state:
    st.session_state.project_name = "New Project"

# Function to reset state
def reset_state():
    st.session_state.data_processor = ESDataProcessor()
    st.session_state.es_calculator = EarnedScheduleCalculator()
    st.session_state.prediction_models = ESPredictionModels()
    st.session_state.has_data = False
    st.session_state.latest_metrics = {}
    st.session_state.forecasts = {}
    st.session_state.best_model = None
    st.experimental_rerun()

# Sidebar with file uploader and manual inputs
with st.sidebar:
    st.header("Data Input")
    
    # Project name
    st.session_state.project_name = st.text_input("Project Name", value=st.session_state.project_name)
    
    # File upload section
    st.subheader("Upload Project Data")
    uploaded_file = st.file_uploader("Upload Excel file with project data", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read and process the Excel file
            processor = st.session_state.data_processor
            
            # Process the uploaded file
            buffer = io.BytesIO(uploaded_file.getvalue())
            processed_data = processor.load_excel(buffer)
            
            # Display upload success
            st.success(f"Successfully loaded data with {len(processed_data)} periods")
            
            # Update session state
            st.session_state.has_data = True
            
            # Validate the data
            is_valid, issues = processor.validate_data()
            if not is_valid:
                st.warning("Data validation found issues:")
                for issue in issues:
                    st.warning(f"- {issue}")
                    
            # Save to database
            conn = get_db_connection()
            cursor = conn.cursor()
            # Add project if new
            cursor.execute('INSERT INTO projects (name) VALUES (?)', (st.session_state.project_name,))
            project_id = cursor.lastrowid
            
            # Add the data
            for idx, row in processed_data.iterrows():
                data_dict = row.to_dict()
                cursor.execute('''
                INSERT INTO project_data 
                (project_id, time_period, pv, ev, ac, es, pd, at, ed) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    project_id,
                    data_dict.get('Time'),
                    data_dict.get('PV'),
                    data_dict.get('EV', None),
                    data_dict.get('AC', None),
                    data_dict.get('ES', None),
                    data_dict.get('PD', None),
                    data_dict.get('AT', None),
                    data_dict.get('ED', None)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Add a separator
    st.markdown("---")
    
    # Manual input section
    st.subheader("Manual Input")
    st.markdown("Enter current project metrics:")
    
    # Two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        manual_at = st.number_input("AT (Actual Time)", min_value=0.0, step=0.1, help="Current time period")
        manual_es = st.number_input("ES (Earned Schedule)", min_value=0.0, step=0.1, help="Current earned schedule")
    
    with col2:
        manual_pd = st.number_input("PD (Planned Duration)", min_value=0.1, step=0.1, help="Total planned duration")
        manual_ed = st.number_input("ED (Earned Duration)", min_value=0.0, step=0.1, help="Current earned duration")
    
    # Additional columns for PV and EV if needed
    manual_pv = st.number_input("PV (Planned Value)", min_value=0.0, step=0.1, help="Current planned value")
    manual_ev = st.number_input("EV (Earned Value)", min_value=0.0, step=0.1, help="Current earned value")
    
    # Button to submit manual input
    submit_manual = st.button("Update with Manual Input")
    
    if submit_manual:
        try:
            # Create a dictionary with manual inputs
            manual_input = {
                'Time': manual_at,
                'AT': manual_at,
                'ES': manual_es,
                'PD': manual_pd,
                'ED': manual_ed
            }
            
            # Add PV and EV if provided
            if manual_pv > 0:
                manual_input['PV'] = manual_pv
            if manual_ev > 0:
                manual_input['EV'] = manual_ev
            
            # If no data loaded yet, create initial dataset
            if not st.session_state.has_data:
                # Create a simple dataframe with one row
                data = pd.DataFrame([manual_input])
                st.session_state.data_processor.processed_data = data
                st.session_state.has_data = True
                st.success("Created initial dataset with manual input")
                
                # Save to database
                conn = get_db_connection()
                cursor = conn.cursor()
                # Add project if new
                cursor.execute('INSERT INTO projects (name) VALUES (?)', (st.session_state.project_name,))
                project_id = cursor.lastrowid
                
                # Add the data
                cursor.execute('''
                INSERT INTO project_data 
                (project_id, time_period, pv, ev, es, pd, at, ed) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    project_id,
                    manual_input.get('Time'),
                    manual_input.get('PV', None),
                    manual_input.get('EV', None),
                    manual_input.get('ES'),
                    manual_input.get('PD'),
                    manual_input.get('AT'),
                    manual_input.get('ED')
                ))
                conn.commit()
                conn.close()
            else:
                # Update existing dataset
                st.session_state.data_processor.update_with_manual_input(manual_input)
                st.success("Updated dataset with manual input")
                
                # Save to database
                conn = get_db_connection()
                cursor = conn.cursor()
                # Find project id
                cursor.execute('SELECT id FROM projects WHERE name = ? ORDER BY created_date DESC LIMIT 1', (st.session_state.project_name,))
                project_id = cursor.fetchone()[0] if cursor.fetchone() else None
                
                if project_id:
                    # Add the data
                    cursor.execute('''
                    INSERT INTO project_data 
                    (project_id, time_period, pv, ev, es, pd, at, ed) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        project_id,
                        manual_input.get('Time'),
                        manual_input.get('PV', None),
                        manual_input.get('EV', None),
                        manual_input.get('ES'),
                        manual_input.get('PD'),
                        manual_input.get('AT'),
                        manual_input.get('ED')
                    ))
                    conn.commit()
                conn.close()
                
        except Exception as e:
            st.error(f"Error updating with manual input: {str(e)}")
            
    # Add a separator
    st.markdown("---")
    
    # Reset button
    if st.button("Reset All Data"):
        reset_state()

# Main content area
if st.session_state.has_data and st.session_state.data_processor is not None:
    try:
        # Process the data and calculate ES metrics
        data = st.session_state.data_processor.processed_data
        latest_metrics = st.session_state.data_processor.get_latest_metrics()
        
        # Update ES Calculator
        es_calc = st.session_state.es_calculator
        es_calc.load_data(data, time_column='Time', pv_column='PV', ev_column='EV', 
                        ac_column='AC' if 'AC' in data.columns else None)
        
        # Set parameters if manually provided
        es_params = {}
        if 'AT' in latest_metrics and not pd.isna(latest_metrics['AT']):
            es_params['at'] = latest_metrics['AT']
        if 'PD' in latest_metrics and not pd.isna(latest_metrics['PD']):
            es_params['pd'] = latest_metrics['PD']
        if 'ES' in latest_metrics and not pd.isna(latest_metrics['ES']):
            es_params['es'] = latest_metrics['ES']
        if 'ED' in latest_metrics and not pd.isna(latest_metrics['ED']):
            es_params['ed'] = latest_metrics['ED']
            
        es_calc.set_parameters(**es_params)
        
        # Calculate all ES metrics
        all_metrics = es_calc.calculate_all_metrics()
        st.session_state.latest_metrics = all_metrics
        
        # Update prediction models with time series data
        # First, prepare the time series data
        if 'ES' not in data.columns or data['ES'].isna().all():
            # Calculate ES for each period if not available
            es_values = []
            for i in range(len(data)):
                period_data = data.iloc[:i+1]
                temp_calc = EarnedScheduleCalculator()
                temp_calc.load_data(period_data, time_column='Time', pv_column='PV', ev_column='EV')
                temp_calc.set_parameters(at=period_data['Time'].iloc[-1])
                es_values.append(temp_calc.calculate_es())
            data['ES'] = es_values
        
        if 'SPI(t)' not in data.columns or data['SPI(t)'].isna().all():
            # Calculate SPI(t) for each period
            spi_t_values = []
            for i in range(len(data)):
                period_data = data.iloc[:i+1]
                temp_calc = EarnedScheduleCalculator()
                temp_calc.load_data(period_data, time_column='Time', pv_column='PV', ev_column='EV')
                temp_calc.set_parameters(at=period_data['Time'].iloc[-1])
                if 'ES' in period_data.columns and not pd.isna(period_data['ES'].iloc[-1]):
                    temp_calc.set_parameters(es=period_data['ES'].iloc[-1])
                spi_t_values.append(temp_calc.calculate_spi_t())
            data['SPI(t)'] = spi_t_values
        
        # Update the prediction models
        pred_models = st.session_state.prediction_models
        pred_models.load_data(data, time_column='Time', spi_t_column='SPI(t)', 
                             es_column='ES', at_column='AT', pd_column='PD')
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["ES Metrics & Dashboard", "Forecasting Models", "Historical Data"])
        
        with tab1:
            st.header("Earned Schedule Metrics Dashboard")
            
            # Display key metrics in cards with 4 columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Earned Schedule (ES)", f"{all_metrics['ES']:.2f}")
                st.metric("Actual Time (AT)", f"{all_metrics['AT']:.2f}")
            
            with col2:
                st.metric("Schedule Performance Index SPI(t)", f"{all_metrics['SPI(t)']:.2f}", 
                         delta=f"{all_metrics['SPI(t)'] - 1:.2f}", 
                         delta_color="normal" if all_metrics['SPI(t)'] >= 1 else "inverse")
                st.metric("Completion % (C = ES/PD)", f"{all_metrics['C']*100:.1f}%")
            
            with col3:
                st.metric("To Complete SPI (TSPI)", f"{all_metrics['TSPI']:.2f}", 
                         delta=f"{all_metrics['TSPI'] - 1:.2f}", 
                         delta_color="inverse")
                st.metric("Recovery Likely", "Yes" if all_metrics['Recovery Likely'] else "No", 
                         delta=None,
                         delta_color="normal" if all_metrics['Recovery Likely'] else "inverse")
            
            with col4:
                st.metric("Estimated Completion (Current)", f"{all_metrics['IEAC(t) - Current Performance']:.2f}", 
                         delta=f"{all_metrics['IEAC(t) - Current Performance'] - all_metrics['PD']:.2f}", 
                         delta_color="inverse" if all_metrics['IEAC(t) - Current Performance'] > all_metrics['PD'] else "normal")
                st.metric("Estimated Completion (PF=1)", f"{all_metrics['IEAC(t) - PF=1']:.2f}", 
                         delta=f"{all_metrics['IEAC(t) - PF=1'] - all_metrics['PD']:.2f}", 
                         delta_color="inverse" if all_metrics['IEAC(t) - PF=1'] > all_metrics['PD'] else "normal")
            
            # Add interpretation text
            st.subheader("Performance Interpretation")
            
            # SPI(t) interpretation
            spi_t = all_metrics['SPI(t)']
            spi_text = """**Schedule Performance Index (time-based)**: """
            if spi_t > 1.05:
                spi_text += f"The project is **ahead of schedule** with SPI(t) = {spi_t:.2f}. It's accomplishing {(spi_t-1)*100:.1f}% more than planned each period."
            elif spi_t >= 0.95:
                spi_text += f"The project is **on schedule** with SPI(t) = {spi_t:.2f}. It's maintaining pace with the plan."
            elif spi_t >= 0.8:
                spi_text += f"The project is **behind schedule** with SPI(t) = {spi_t:.2f}. It's accomplishing {(1-spi_t)*100:.1f}% less than planned each period."
            else:
                spi_text += f"The project is **significantly behind schedule** with SPI(t) = {spi_t:.2f}. It's accomplishing {(1-spi_t)*100:.1f}% less than planned each period."
            
            # TSPI interpretation
            tspi = all_metrics['TSPI']
            tspi_text = """**To Complete Schedule Performance Index**: """
            if tspi <= 1.0:
                tspi_text += f"With TSPI = {tspi:.2f}, the project can complete on time with current or even slightly lower performance."
            elif tspi <= 1.05:
                tspi_text += f"With TSPI = {tspi:.2f}, the project needs a modest {(tspi-1)*100:.1f}% improvement in schedule performance to finish on time."
            elif tspi <= 1.1:
                tspi_text += f"With TSPI = {tspi:.2f}, the project needs a significant {(tspi-1)*100:.1f}% improvement in schedule performance to finish on time. This may be challenging but achievable."
            else:
                tspi_text += f"With TSPI = {tspi:.2f}, exceeding the critical threshold of 1.10, recovery to the original schedule is very unlikely without scope changes or re-baselining."
            
            # Combine interpretations
            st.markdown(f"{spi_text}\n\n{tspi_text}")
            
            # Forecast interpretation
            st.markdown(f"""**Forecast Completion**:  
            Based on current performance, the project is estimated to complete at period **{all_metrics['IEAC(t) - Current Performance']:.2f}** vs planned duration of {all_metrics['PD']:.2f}, 
            resulting in a {'delay' if all_metrics['IEAC(t) - Current Performance'] > all_metrics['PD'] else 'gain'} of 
            {abs(all_metrics['IEAC(t) - Current Performance'] - all_metrics['PD']):.2f} periods.""")
            
            # S-Curve Visualization
            st.subheader("S-Curve Analysis")
            
            # Prepare data for S-curve
            # For now, we'll just plot PV and EV over time
            fig_s_curve = go.Figure()
            
            # Add Planned Value (PV) curve
            fig_s_curve.add_trace(go.Scatter(
                x=data['Time'],
                y=data['PV'],
                mode='lines+markers',
                name='Planned Value (PV)',
                line=dict(color='#FFC000', width=2)
            ))
            
            # Add Earned Value (EV) curve
            fig_s_curve.add_trace(go.Scatter(
                x=data['Time'],
                y=data['EV'],
                mode='lines+markers',
                name='Earned Value (EV)',
                line=dict(color='#FF6600', width=2)
            ))
            
            # Add Actual Cost (AC) if available
            if 'AC' in data.columns and not data['AC'].isna().all():
                fig_s_curve.add_trace(go.Scatter(
                    x=data['Time'],
                    y=data['AC'],
                    mode='lines+markers',
                    name='Actual Cost (AC)',
                    line=dict(color='#0070C0', width=2)
                ))
            
            # Update layout
            fig_s_curve.update_layout(
                title='Project S-Curve: Planned vs Actual Progress',
                xaxis_title='Time Period',
                yaxis_title='Value',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(fig_s_curve, use_container_width=True)
            
            # Add ES and SPI(t) visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # ES over time
                fig_es = go.Figure()
                
                # Add ES curve
                fig_es.add_trace(go.Scatter(
                    x=data['Time'],
                    y=data['ES'],
                    mode='lines+markers',
                    name='Earned Schedule (ES)',
                    line=dict(color='#00B050', width=2)
                ))
                
                # Add reference line (ES = AT, on schedule)
                fig_es.add_trace(go.Scatter(
                    x=data['Time'],
                    y=data['Time'],
                    mode='lines',
                    name='On Schedule (ES = AT)',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # Update layout
                fig_es.update_layout(
                    title='Earned Schedule vs Actual Time',
                    xaxis_title='Actual Time (AT)',
                    yaxis_title='Earned Schedule (ES)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig_es, use_container_width=True)
            
            with col2:
                # SPI(t) over time
                fig_spi = go.Figure()
                
                # Add SPI(t) curve
                fig_spi.add_trace(go.Scatter(
                    x=data['Time'],
                    y=data['SPI(t)'],
                    mode='lines+markers',
                    name='SPI(t)',
                    line=dict(color='#7030A0', width=2)
                ))
                
                # Add reference line (SPI(t) = 1, on schedule)
                fig_spi.add_trace(go.Scatter(
                    x=[data['Time'].min(), data['Time'].max()],
                    y=[1, 1],
                    mode='lines',
                    name='On Schedule (SPI(t) = 1)',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # Update layout
                fig_spi.update_layout(
                    title='Schedule Performance Index (time) Trend',
                    xaxis_title='Time Period',
                    yaxis_title='SPI(t)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig_spi, use_container_width=True)
        
        with tab2:
            st.header("Forecasting Models")
            
            # Prediction models section
            st.subheader("Schedule Forecast Analysis")
            
            # Fit different prediction models
            model_selection = st.multiselect(
                "Select models to apply",
                ["Linear Regression", "Exponential Smoothing", "ARIMA", "Random Forest", "Gradient Boosting"],
                default=["Linear Regression", "Exponential Smoothing", "ARIMA"]
            )
            
            # Forecast parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_periods = st.number_input("Forecast Periods", min_value=1, value=5, step=1)
            with col2:
                target_variable = st.selectbox("Target Variable", ["Earned Schedule (ES)", "Schedule Performance Index (SPI(t))"], index=0)
            with col3:
                confidence_level = st.slider("Confidence Level", min_value=50, max_value=99, value=90, step=5)
                
            run_forecast = st.button("Run Forecast Models")
            
            if run_forecast:
                # Clear previous forecasts
                st.session_state.forecasts = {}
                
                # Set the target variable
                target = 'es' if target_variable == "Earned Schedule (ES)" else 'spi_t'
                
                # Run selected models
                with st.spinner("Running forecast models..."):
                    if "Linear Regression" in model_selection:
                        try:
                            st.session_state.forecasts['linear_regression'] = pred_models.fit_linear_regression(target=target, forecast_periods=forecast_periods)
                        except Exception as e:
                            st.error(f"Error in Linear Regression model: {str(e)}")
                            
                    if "Exponential Smoothing" in model_selection:
                        try:
                            st.session_state.forecasts['exp_smoothing'] = pred_models.fit_exponential_smoothing(target=target, forecast_periods=forecast_periods)
                        except Exception as e:
                            st.error(f"Error in Exponential Smoothing model: {str(e)}")
                            
                    if "ARIMA" in model_selection:
                        try:
                            st.session_state.forecasts['arima'] = pred_models.fit_arima(target=target, forecast_periods=forecast_periods)
                        except Exception as e:
                            st.error(f"Error in ARIMA model: {str(e)}")
                            
                    if "Random Forest" in model_selection:
                        try:
                            st.session_state.forecasts['random_forest'] = pred_models.fit_machine_learning(
                                model_type='random_forest', target=target, forecast_periods=forecast_periods
                            )
                        except Exception as e:
                            st.error(f"Error in Random Forest model: {str(e)}")
                            
                    if "Gradient Boosting" in model_selection:
                        try:
                            st.session_state.forecasts['gradient_boosting'] = pred_models.fit_machine_learning(
                                model_type='gradient_boosting', target=target, forecast_periods=forecast_periods
                            )
                        except Exception as e:
                            st.error(f"Error in Gradient Boosting model: {str(e)}")
                    
                    # Find best model
                    try:
                        st.session_state.best_model = pred_models.find_best_model(metric='mae', target=target)
                    except Exception as e:
                        st.warning(f"Error finding best model: {str(e)}")
                
                st.success("Forecast models completed!")
            
            # Display forecasts if available
            if st.session_state.forecasts:
                # Plot forecast
                st.subheader("Forecast Visualization")
                
                # Get forecast from each model and plot
                fig = go.Figure()
                
                # Add actual data
                target_col = 'ES' if target_variable == "Earned Schedule (ES)" else 'SPI(t)'
                
                fig.add_trace(go.Scatter(
                    x=data['Time'],
                    y=data[target_col],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='black', width=2)
                ))
                
                # Define the visualization colors for each model
                model_colors = {
                    'linear_regression': '#FF6600',
                    'exp_smoothing': '#00B050',
                    'arima': '#0070C0',
                    'random_forest': '#7030A0',
                    'gradient_boosting': '#D9534F'
                }
                
                # Add forecast from each model
                for model_key, forecast in st.session_state.forecasts.items():
                    # Format the model name for display
                    model_name = model_key.replace('_', ' ').title()
                    
                    # Get the color
                    color = model_colors.get(model_key, '#AAAAAA')
                    
                    # Add the forecast line
                    fig.add_trace(go.Scatter(
                        x=forecast['future_times'],
                        y=forecast['future_values'],
                        mode='lines',
                        name=f"{model_name} Forecast",
                        line=dict(color=color, width=2, dash='dash')
                    ))
                
                # Add PD reference line if target is ES
                if target_variable == "Earned Schedule (ES)" and 'PD' in all_metrics:
                    fig.add_trace(go.Scatter(
                        x=[data['Time'].min(), data['Time'].max() + forecast_periods],
                        y=[all_metrics['PD'], all_metrics['PD']],
                        mode='lines',
                        name='Planned Duration (PD)',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                
                # Add SPI(t)=1 reference line if target is SPI(t)
                elif target_variable == "Schedule Performance Index (SPI(t))":
                    fig.add_trace(go.Scatter(
                        x=[data['Time'].min(), data['Time'].max() + forecast_periods],
                        y=[1, 1],
                        mode='lines',
                        name='On Schedule (SPI(t) = 1)',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f'{target_variable} Forecast',
                    xaxis_title='Time Period',
                    yaxis_title=target_variable,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show completion date forecasts if target is ES
                if target_variable == "Earned Schedule (ES)":
                    st.subheader("Completion Date Forecasts")
                    
                    # Get completion forecasts
                    completion_forecasts = pred_models.get_all_completion_forecasts(pd=all_metrics['PD'])
                    
                    # Create a dataframe to display the results
                    forecast_data = []
                    for model_key, (completion_time, confidence) in completion_forecasts.items():
                        if completion_time is not None:
                            delay = completion_time - all_metrics['PD']
                            forecast_data.append({
                                'Model': model_key.replace('_', ' ').title(),
                                'Completion Time': f"{completion_time:.2f}",
                                'Confidence Range': f"±{confidence:.2f}",
                                'Delay/Early': f"{delay:.2f}",
                                'Status': "Early" if delay < 0 else "On Time" if delay == 0 else "Delay"
                            })
                    
                    if forecast_data:
                        forecast_df = pd.DataFrame(forecast_data)
                        # Color the Status column
                        status_colors = {
                            'Early': 'background-color: #d4edda; color: #155724',
                            'On Time': 'background-color: #d1ecf1; color: #0c5460',
                            'Delay': 'background-color: #f8d7da; color: #721c24'
                        }
                        
                        # Apply the color formatting
                        styled_df = forecast_df.style.applymap(
                            lambda v: status_colors.get(v, ''),
                            subset=['Status']
                        )
                        
                        st.table(styled_df)
                        
                        # Highlight the recommended model
                        if st.session_state.best_model:
                            best_model_name = st.session_state.best_model.replace('_', ' ').title()
                            st.info(f"**Recommended forecast model:** {best_model_name} (lowest prediction error)")
                            
                            # Get specific completion prediction from best model
                            best_completion, best_confidence = completion_forecasts[st.session_state.best_model]
                            if best_completion is not None:
                                delay = best_completion - all_metrics['PD']
                                status = "ahead of schedule" if delay < 0 else "on schedule" if delay == 0 else "behind schedule"
                                
                                st.markdown(f"""Based on the {best_model_name} model, the project is expected to complete 
                                at period **{best_completion:.2f}** {'±' if best_confidence else ''}{best_confidence:.2f if best_confidence else ''}, 
                                making it **{abs(delay):.2f}** periods {status}.
                                """)
                    else:
                        st.warning("Could not determine completion forecasts from the models.")
                
                # Show model metrics
                st.subheader("Model Performance Metrics")
                
                # Create a dataframe to display the model metrics
                metrics_data = []
                for model_key, metrics in pred_models.metrics.items():
                    if model_key in st.session_state.forecasts:
                        metrics_data.append({
                            'Model': model_key.replace('_', ' ').title(),
                            'MAE': f"{metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else 'N/A',
                            'MSE': f"{metrics.get('mse', 'N/A'):.4f}" if isinstance(metrics.get('mse'), (int, float)) else 'N/A',
                            'R²': f"{metrics.get('r2', 'N/A'):.4f}" if isinstance(metrics.get('r2'), (int, float)) else 'N/A',
                            'Notes': getattr(pred_models, f'_model_notes_{model_key}', 'N/A') if hasattr(pred_models, f'_model_notes_{model_key}') else ''
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.table(metrics_df)
                    
                    st.markdown("""**Notes on metrics:**
                    - **MAE**: Mean Absolute Error - lower is better
                    - **MSE**: Mean Squared Error - lower is better
                    - **R²**: Coefficient of determination - higher is better (closer to 1.0)""")
        
        with tab3:
            st.header("Historical Project Data")
            
            # Show the raw data table
            st.dataframe(data)
            
            # Option to download the data
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ES_Project_{st.session_state.project_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please try refreshing the page or uploading your data again.")
else:
    # No data yet, show instructions
    st.info("Please upload project data using the sidebar or enter manual inputs to get started.")
    
    # Show some example instructions
    st.markdown("""
    ### Getting Started with ES-Agent:
    
    1. **Upload your project data** using the file uploader in the sidebar
       - Prepare an Excel file with Time, PV, and EV columns
       - Optionally include AC for cost analysis
       
    2. **Or enter data manually** using the form in the sidebar
       - AT (Actual Time): Current time period
       - ES (Earned Schedule): Current earned schedule value
       - PD (Planned Duration): Total planned duration
       - Other metrics as available
       
    3. **View the analysis** once data is loaded
       - ES Dashboard: Key performance indicators
       - Forecasting: Schedule predictions
       - Historical Data: Raw data table
    """)
    
    # Display a sample visualization to help users understand
    try:
        st.subheader("Sample Visualization (for reference)")
        # Generate sample data
        sample_time = list(range(1, 11))
        sample_pv = [10*t for t in sample_time]
        sample_ev = [8*t for t in sample_time]
        
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
        
        st.caption("This is a sample visualization. Upload your data to see actual project performance.")
    except Exception as e:
        # If visualization fails, just show a message
        st.write("Sample visualization could not be displayed.")
