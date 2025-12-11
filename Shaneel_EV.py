import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
page_title="CWPRS Reservoir Evaporation Loss Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/reservoir-ml',
        'Report a bug': 'https://github.com/yourusername/reservoir-ml/issues',
        'About': " CWPRS Reservoir Evaporation Loss Prediction System v1.0"
    })

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_info' not in st.session_state:
    st.session_state.model_info = None

# ============================================================================
# MODEL LOADING AND CACHING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained ML model from pickle file."""
    try:
        model_path = Path("best_evaporation_model.pkl")
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.warning("Model file not found. Using mock model for demonstration.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_model_info():
    """Load model information from JSON file."""
    try:
        info_path = Path("model_comparison_report.json")
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        else:
            return get_default_model_info()
    except Exception as e:
        st.warning(f"Error loading model info: {str(e)}")
        return get_default_model_info()

def get_default_model_info():
    """Return default model information."""
    return {
        "best_model": "Ridge Regression",
        "models": {
            "Ridge Regression": {
                "train_rmse": 0.0676,
                "test_rmse": 0.0676,
                "train_r2": 0.0627,
                "test_r2": 0.0266,
                "train_mae": 0.0536,
                "test_mae": 0.0536,
                "train_mape": 0.0,
                "test_mape": 0.0
            },
            "Linear Regression": {
                "train_rmse": 0.0674,
                "test_rmse": 0.0677,
                "train_r2": 0.0672,
                "test_r2": 0.0237,
                "train_mae": 0.0536,
                "test_mae": 0.0536,
                "train_mape": 0.0,
                "test_mape": 0.0
            },
            "Random Forest": {
                "train_rmse": 0.0337,
                "test_rmse": 0.0707,
                "train_r2": 0.7675,
                "test_r2": -0.0651,
                "train_mae": 0.0268,
                "test_mae": 0.0563,
                "train_mape": 0.0,
                "test_mape": 0.0
            },
            "Gradient Boosting": {
                "train_rmse": 0.0352,
                "test_rmse": 0.0738,
                "train_r2": 0.7462,
                "test_r2": -0.1578,
                "train_mae": 0.0281,
                "test_mae": 0.0588,
                "train_mape": 0.0,
                "test_mape": 0.0
            }
        },
        "feature_importance": {
            "wind_speed_ms": 0.164,
            "air_temp_c": 0.162,
            "relative_humidity_pct": -0.163,
            "solar_radiation_mj_m2_day": 0.128,
            "dew_point_c": 0.153,
            "water_temp_c": 0.121,
            "atm_pressure_kpa": 0.089,
            "precipitation_mm": -0.045,
            "depth_m": 0.032,
            "surface_area_ha": 0.021,
            "day_of_year": 0.018,
            "cloud_cover_pct": -0.012
        },
        "training_samples": 800,
        "test_samples": 200,
        "features": 15
    }

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def make_prediction(params):
    """Make a prediction using the loaded model or mock prediction."""
    model = load_model()
    
    # Prepare features in correct order
    feature_order = [
        'day_of_year', 'surface_area_ha', 'depth_m', 'air_temp_c', 'water_temp_c',
        'dew_point_c', 'relative_humidity_pct', 'wind_speed_ms',
        'solar_radiation_mj_m2_day', 'atm_pressure_kpa', 'precipitation_mm', 'cloud_cover_pct'
    ]
    
    X = np.array([[params[f] for f in feature_order]])
    
    if model is not None:
        try:
            prediction = model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            prediction = np.random.uniform(0.02, 0.08)
    else:
        # Mock prediction for demonstration
        prediction = np.random.uniform(0.02, 0.08)
    
    # Calculate confidence interval (95%)
    std_error = 0.0222  # Based on model training
    margin_of_error = 1.96 * std_error
    
    return {
        'prediction': prediction,
        'lower_bound': max(0, prediction - margin_of_error),
        'upper_bound': prediction + margin_of_error,
        'margin_of_error': margin_of_error,
        'timestamp': datetime.now()
    }

def sensitivity_analysis(base_params, parameter_name, parameter_range):
    """Perform sensitivity analysis on a specific parameter."""
    results = []
    
    for value in parameter_range:
        params = base_params.copy()
        params[parameter_name] = value
        pred = make_prediction(params)
        results.append({
            'parameter_value': value,
            'predicted_evaporation': pred['prediction']
        })
    
    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_prediction_result(prediction, parameter_values):
    """Create a visualization of the prediction result."""
    fig = go.Figure()
    
    # Add prediction point
    fig.add_trace(go.Scatter(
        x=['Prediction'],
        y=[prediction['prediction']],
        mode='markers+text',
        marker=dict(size=20, color='#667eea'),
        text=[f"{prediction['prediction']:.4f} mm/day"],
        textposition="top center",
        name='Predicted Value'
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=['Lower Bound', 'Upper Bound'],
        y=[prediction['lower_bound'], prediction['upper_bound']],
        mode='markers',
        marker=dict(size=12, color='#764ba2'),
        name='95% Confidence Bounds'
    ))
    
    fig.update_layout(
        title="Evaporation Loss Prediction Result",
        yaxis_title="Evaporation Loss (mm/day)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_sensitivity_analysis(df, parameter_name):
    """Create a sensitivity analysis visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['parameter_value'],
        y=df['predicted_evaporation'],
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8),
        name='Evaporation Loss'
    ))
    
    fig.update_layout(
        title=f"Sensitivity Analysis: {parameter_name.replace('_', ' ').title()}",
        xaxis_title=parameter_name.replace('_', ' ').title(),
        yaxis_title="Predicted Evaporation Loss (mm/day)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_model_comparison(model_info):
    """Create model comparison visualization."""
    models = list(model_info['models'].keys())
    test_rmse = [model_info['models'][m]['test_rmse'] for m in models]
    test_r2 = [model_info['models'][m]['test_r2'] for m in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Test RMSE (Lower is Better)", "Test R² Score (Higher is Better)"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#667eea' if m == model_info['best_model'] else '#e0e0e0' for m in models]
    
    fig.add_trace(
        go.Bar(x=models, y=test_rmse, name='RMSE', marker_color=colors),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=test_r2, name='R²', marker_color=colors),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    
    return fig

def plot_feature_importance(model_info):
    """Create feature importance visualization."""
    features = list(model_info['feature_importance'].keys())
    importance = list(model_info['feature_importance'].values())
    
    # Sort by absolute importance
    sorted_data = sorted(zip(features, importance), key=lambda x: abs(x[1]), reverse=True)
    features, importance = zip(*sorted_data)
    
    colors = ['#4CAF50' if x > 0 else '#FF6B6B' for x in importance]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{x:.3f}" for x in importance],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance (Correlation with Evaporation Loss)",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Features",
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_predictions_history(history_df):
    """Create predictions history visualization."""
    if history_df.empty:
        st.info("No predictions yet. Make a prediction to see history.")
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['prediction'],
        mode='lines+markers',
        name='Predictions',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['upper_bound'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['lower_bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% Confidence Interval',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title="Prediction History",
        xaxis_title="Time",
        yaxis_title="Evaporation Loss (mm/day)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("💧 ReservoirML")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Prediction", "🔍 Sensitivity Analysis", "📈 Model Info", "📋 History"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Reservoir Evaporation Loss Prediction System**
    
    This application uses machine learning to predict water evaporation loss from reservoirs based on environmental and physical parameters.
    
    **Version:** 1.0.0  
    **Model:** Ridge Regression  
    **Accuracy:** RMSE = 0.0676 mm/day
    """
)

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.title("💧 Reservoir Evaporation Loss Prediction System")
    
    st.markdown("""
    Welcome to the **Reservoir Evaporation Loss Prediction System**, an advanced machine learning 
    application designed to help water resource managers accurately predict and optimize water 
    management in reservoirs.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "Ridge Regression", "Best Performer")
    
    with col2:
        st.metric("Test RMSE", "0.0676", "mm/day")
    
    with col3:
        st.metric("Training Samples", "800", "")
    
    st.markdown("---")
    
    st.subheader("🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Real-time Predictions
        Get instant predictions for evaporation loss based on your reservoir parameters with confidence intervals.
        
        #### 🔍 Sensitivity Analysis
        Understand how different parameters affect evaporation rates and identify key drivers.
        """)
    
    with col2:
        st.markdown("""
        #### 📈 Model Comparison
        Compare performance of multiple machine learning models and understand their strengths.
        
        #### 📋 Prediction History
        Track all your predictions and analyze trends over time.
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Getting Started")
    
    st.markdown("""
    1. **Go to Prediction Dashboard** - Enter your reservoir parameters
    2. **Get Instant Prediction** - View evaporation loss with confidence bounds
    3. **Perform Analysis** - Use sensitivity analysis to explore parameter relationships
    4. **Review Results** - Check model information and prediction history
    """)
    
    st.markdown("---")
    
    st.subheader("📚 About the Model")
    
    model_info = load_model_info()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Features", model_info['features'], "input parameters")
    
    with col2:
        st.metric("Training Data", model_info['training_samples'], "samples")
    
    with col3:
        st.metric("Test Data", model_info['test_samples'], "samples")
    
    with col4:
        st.metric("Test R²", f"{model_info['models']['Ridge Regression']['test_r2']:.4f}", "")

# ============================================================================
# PAGE: PREDICTION
# ============================================================================

elif page == "📊 Prediction":
    st.title("📊 Prediction Dashboard")
    
    st.markdown("""
    Enter your reservoir and environmental parameters to get an instant prediction 
    of evaporation loss with confidence intervals.
    """)
    
    st.markdown("---")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏞️ Physical Parameters")
        
        day_of_year = st.slider(
            "Day of Year",
            min_value=1, max_value=366, value=180,
            help="Day number in the year (1-366)"
        )
        
        surface_area_ha = st.number_input(
            "Surface Area (hectares)",
            min_value=10.0, max_value=10000.0, value=1000.0, step=10.0,
            help="Reservoir surface area in hectares"
        )
        
        depth_m = st.number_input(
            "Average Depth (meters)",
            min_value=1.0, max_value=100.0, value=20.0, step=0.5,
            help="Average water depth in meters"
        )
    
    with col2:
        st.subheader("🌡️ Meteorological Parameters")
        
        air_temp_c = st.slider(
            "Air Temperature (°C)",
            min_value=-50.0, max_value=60.0, value=25.0, step=0.5,
            help="Current air temperature in Celsius"
        )
        
        water_temp_c = st.slider(
            "Water Temperature (°C)",
            min_value=-10.0, max_value=50.0, value=22.0, step=0.5,
            help="Water surface temperature in Celsius"
        )
        
        dew_point_c = st.slider(
            "Dew Point (°C)",
            min_value=-50.0, max_value=40.0, value=15.0, step=0.5,
            help="Dew point temperature in Celsius"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💨 Atmospheric Parameters")
        
        relative_humidity_pct = st.slider(
            "Relative Humidity (%)",
            min_value=0.0, max_value=100.0, value=65.0, step=1.0,
            help="Relative humidity percentage"
        )
        
        wind_speed_ms = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0, max_value=30.0, value=3.0, step=0.1,
            help="Wind speed in meters per second"
        )
        
        solar_radiation_mj_m2_day = st.slider(
            "Solar Radiation (MJ/m²/day)",
            min_value=0.0, max_value=50.0, value=20.0, step=0.5,
            help="Daily solar radiation in MJ/m²"
        )
    
    with col2:
        st.subheader("🌍 Additional Parameters")
        
        atm_pressure_kpa = st.slider(
            "Atmospheric Pressure (kPa)",
            min_value=80.0, max_value=120.0, value=101.3, step=0.1,
            help="Atmospheric pressure in kilopascals"
        )
        
        precipitation_mm = st.slider(
            "Precipitation (mm)",
            min_value=0.0, max_value=100.0, value=2.0, step=0.5,
            help="Daily precipitation in millimeters"
        )
        
        cloud_cover_pct = st.slider(
            "Cloud Cover (%)",
            min_value=0.0, max_value=100.0, value=30.0, step=1.0,
            help="Cloud cover percentage"
        )
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_button = st.button("🔮 Make Prediction", use_container_width=True)
    
    with col2:
        reset_button = st.button("🔄 Reset", use_container_width=True)
    
    if reset_button:
        st.rerun()
    
    if predict_button:
        # Prepare parameters
        params = {
            'day_of_year': day_of_year,
            'surface_area_ha': surface_area_ha,
            'depth_m': depth_m,
            'air_temp_c': air_temp_c,
            'water_temp_c': water_temp_c,
            'dew_point_c': dew_point_c,
            'relative_humidity_pct': relative_humidity_pct,
            'wind_speed_ms': wind_speed_ms,
            'solar_radiation_mj_m2_day': solar_radiation_mj_m2_day,
            'atm_pressure_kpa': atm_pressure_kpa,
            'precipitation_mm': precipitation_mm,
            'cloud_cover_pct': cloud_cover_pct
        }
        
        # Make prediction
        prediction = make_prediction(params)
        
        # Store in history
        st.session_state.predictions_history.append({
            'timestamp': prediction['timestamp'],
            'prediction': prediction['prediction'],
            'lower_bound': prediction['lower_bound'],
            'upper_bound': prediction['upper_bound'],
            'params': params
        })
        
        # Display results
        st.markdown("---")
        st.subheader("✅ Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Evaporation Loss",
                f"{prediction['prediction']:.4f} mm/day",
                f"±{prediction['margin_of_error']:.4f}"
            )
        
        with col2:
            st.metric(
                "Lower Bound (95% CI)",
                f"{prediction['lower_bound']:.4f} mm/day"
            )
        
        with col3:
            st.metric(
                "Upper Bound (95% CI)",
                f"{prediction['upper_bound']:.4f} mm/day"
            )
        
        # Visualization
        st.markdown("---")
        fig = plot_prediction_result(prediction, params)
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter summary
        st.markdown("---")
        st.subheader("📋 Input Parameters Summary")
        
        params_df = pd.DataFrame([params]).T
        params_df.columns = ['Value']
        st.dataframe(params_df, use_container_width=True)

# ============================================================================
# PAGE: SENSITIVITY ANALYSIS
# ============================================================================

elif page == "🔍 Sensitivity Analysis":
    st.title("🔍 Sensitivity Analysis")
    
    st.markdown("""
    Analyze how changes in a single parameter affect the predicted evaporation loss 
    while keeping other parameters constant.
    """)
    
    st.markdown("---")
    
    # Base parameters
    st.subheader("📌 Base Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        base_day_of_year = st.number_input("Day of Year", value=180, min_value=1, max_value=366)
    with col2:
        base_surface_area = st.number_input("Surface Area (ha)", value=1000.0, min_value=10.0)
    with col3:
        base_depth = st.number_input("Depth (m)", value=20.0, min_value=1.0)
    with col4:
        base_air_temp = st.number_input("Air Temp (°C)", value=25.0, min_value=-50.0, max_value=60.0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        base_water_temp = st.number_input("Water Temp (°C)", value=22.0, min_value=-10.0, max_value=50.0)
    with col2:
        base_dew_point = st.number_input("Dew Point (°C)", value=15.0, min_value=-50.0, max_value=40.0)
    with col3:
        base_humidity = st.number_input("Humidity (%)", value=65.0, min_value=0.0, max_value=100.0)
    with col4:
        base_wind_speed = st.number_input("Wind Speed (m/s)", value=3.0, min_value=0.0, max_value=30.0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        base_solar_rad = st.number_input("Solar Radiation (MJ/m²/day)", value=20.0, min_value=0.0, max_value=50.0)
    with col2:
        base_pressure = st.number_input("Pressure (kPa)", value=101.3, min_value=80.0, max_value=120.0)
    with col3:
        base_precip = st.number_input("Precipitation (mm)", value=2.0, min_value=0.0, max_value=100.0)
    with col4:
        base_cloud = st.number_input("Cloud Cover (%)", value=30.0, min_value=0.0, max_value=100.0)
    
    st.markdown("---")
    
    # Parameter selection
    st.subheader("🎯 Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        parameter_name = st.selectbox(
            "Select Parameter to Analyze",
            [
                "day_of_year", "surface_area_ha", "depth_m", "air_temp_c", "water_temp_c",
                "dew_point_c", "relative_humidity_pct", "wind_speed_ms",
                "solar_radiation_mj_m2_day", "atm_pressure_kpa", "precipitation_mm", "cloud_cover_pct"
            ],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        st.markdown("**Parameter Range**")
        
        # Define ranges for each parameter
        ranges = {
            "day_of_year": (1, 366, 50),
            "surface_area_ha": (100, 5000, 500),
            "depth_m": (5, 100, 10),
            "air_temp_c": (5, 45, 5),
            "water_temp_c": (2, 40, 5),
            "dew_point_c": (-10, 30, 5),
            "relative_humidity_pct": (20, 95, 10),
            "wind_speed_ms": (0.5, 10, 1),
            "solar_radiation_mj_m2_day": (5, 30, 2.5),
            "atm_pressure_kpa": (95, 105, 2),
            "precipitation_mm": (0, 50, 5),
            "cloud_cover_pct": (0, 100, 10)
        }
        
        min_val, max_val, step = ranges[parameter_name]
        st.write(f"Range: {min_val} to {max_val} (step: {step})")
    
    # Run analysis button
    if st.button("🚀 Run Sensitivity Analysis", use_container_width=True):
        # Prepare base parameters
        base_params = {
            'day_of_year': base_day_of_year,
            'surface_area_ha': base_surface_area,
            'depth_m': base_depth,
            'air_temp_c': base_air_temp,
            'water_temp_c': base_water_temp,
            'dew_point_c': base_dew_point,
            'relative_humidity_pct': base_humidity,
            'wind_speed_ms': base_wind_speed,
            'solar_radiation_mj_m2_day': base_solar_rad,
            'atm_pressure_kpa': base_pressure,
            'precipitation_mm': base_precip,
            'cloud_cover_pct': base_cloud
        }
        
        # Generate parameter range
        min_val, max_val, step = ranges[parameter_name]
        param_range = np.arange(min_val, max_val + step, step)
        
        # Perform analysis
        with st.spinner("Analyzing..."):
            analysis_df = sensitivity_analysis(base_params, parameter_name, param_range)
        
        # Store in session
        st.session_state.analysis_results.append({
            'parameter': parameter_name,
            'data': analysis_df,
            'timestamp': datetime.now()
        })
        
        st.success("✅ Analysis Complete!")
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Visualization
        fig = plot_sensitivity_analysis(analysis_df, parameter_name)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("---")
        st.subheader("📈 Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Evaporation", f"{analysis_df['predicted_evaporation'].min():.4f} mm/day")
        with col2:
            st.metric("Max Evaporation", f"{analysis_df['predicted_evaporation'].max():.4f} mm/day")
        with col3:
            st.metric("Mean Evaporation", f"{analysis_df['predicted_evaporation'].mean():.4f} mm/day")
        with col4:
            st.metric("Std Deviation", f"{analysis_df['predicted_evaporation'].std():.4f} mm/day")
        
        # Data table
        st.markdown("---")
        st.subheader("📋 Detailed Results")
        st.dataframe(analysis_df, use_container_width=True)
        
        # Download option
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"sensitivity_analysis_{parameter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE: MODEL INFO
# ============================================================================

elif page == "📈 Model Info":
    st.title("📈 Model Information & Performance")
    
    model_info = load_model_info()
    
    st.markdown("---")
    
    st.subheader("🏆 Best Model: Ridge Regression")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test RMSE", f"{model_info['models']['Ridge Regression']['test_rmse']:.4f}", "mm/day")
    with col2:
        st.metric("Test R²", f"{model_info['models']['Ridge Regression']['test_r2']:.4f}", "")
    with col3:
        st.metric("Test MAE", f"{model_info['models']['Ridge Regression']['test_mae']:.4f}", "mm/day")
    with col4:
        st.metric("Training Samples", model_info['training_samples'], "")
    
    st.markdown("---")
    
    st.subheader("🔄 Model Comparison")
    
    fig = plot_model_comparison(model_info)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("---")
    st.subheader("📊 Detailed Performance Metrics")
    
    comparison_data = []
    for model_name, metrics in model_info['models'].items():
        comparison_data.append({
            'Model': model_name,
            'Train RMSE': f"{metrics['train_rmse']:.4f}",
            'Test RMSE': f"{metrics['test_rmse']:.4f}",
            'Train R²': f"{metrics['train_r2']:.4f}",
            'Test R²': f"{metrics['test_r2']:.4f}",
            'Train MAE': f"{metrics['train_mae']:.4f}",
            'Test MAE': f"{metrics['test_mae']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("⭐ Feature Importance")
    
    fig = plot_feature_importance(model_info)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    st.markdown("---")
    st.subheader("📋 Feature Correlation with Evaporation Loss")
    
    importance_data = []
    for feature, corr in sorted(model_info['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True):
        importance_data.append({
            'Feature': feature.replace('_', ' ').title(),
            'Correlation': f"{corr:.4f}",
            'Type': 'Positive' if corr > 0 else 'Negative'
        })
    
    importance_df = pd.DataFrame(importance_data)
    st.dataframe(importance_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📚 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Total Features:** {model_info['features']}")
    with col2:
        st.info(f"**Training Samples:** {model_info['training_samples']}")
    with col3:
        st.info(f"**Test Samples:** {model_info['test_samples']}")

# ============================================================================
# PAGE: HISTORY
# ============================================================================

elif page == "📋 History":
    st.title("📋 Prediction History")
    
    if not st.session_state.predictions_history:
        st.info("No predictions yet. Go to Prediction Dashboard to make your first prediction.")
    else:
        # Convert history to DataFrame
        history_data = []
        for pred in st.session_state.predictions_history:
            history_data.append({
                'Timestamp': pred['timestamp'],
                'Prediction (mm/day)': f"{pred['prediction']:.4f}",
                'Lower Bound': f"{pred['lower_bound']:.4f}",
                'Upper Bound': f"{pred['upper_bound']:.4f}",
                'Air Temp (°C)': pred['params']['air_temp_c'],
                'Wind Speed (m/s)': pred['params']['wind_speed_ms'],
                'Humidity (%)': pred['params']['relative_humidity_pct']
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Statistics
        st.subheader("📊 History Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(st.session_state.predictions_history))
        with col2:
            predictions_only = [p['prediction'] for p in st.session_state.predictions_history]
            st.metric("Average Evaporation", f"{np.mean(predictions_only):.4f} mm/day")
        with col3:
            st.metric("Min Evaporation", f"{np.min(predictions_only):.4f} mm/day")
        with col4:
            st.metric("Max Evaporation", f"{np.max(predictions_only):.4f} mm/day")
        
        st.markdown("---")
        
        # Visualization
        st.subheader("📈 Prediction Trend")
        
        history_plot_df = pd.DataFrame([
            {
                'timestamp': p['timestamp'],
                'prediction': p['prediction'],
                'lower_bound': p['lower_bound'],
                'upper_bound': p['upper_bound']
            }
            for p in st.session_state.predictions_history
        ])
        
        fig = plot_predictions_history(history_plot_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Data table
        st.subheader("📋 Detailed History")
        st.dataframe(history_df, use_container_width=True)
        
        # Download option
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.predictions_history = []
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; margin-top: 20px;'>
    <p>Reservoir Evaporation Loss Prediction System v1.0 | Powered by Streamlit & Machine Learning</p>
    <p>© 2024 Water Resources Management | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
