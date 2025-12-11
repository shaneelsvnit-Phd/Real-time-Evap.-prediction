import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CWPRS: Evaporation AI", page_icon="💧", layout="wide")

# Custom CSS (From newEVapp.py)
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #eef2f7; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* CARD STYLING */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white; border-radius: 15px; padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* DARK SIDEBAR STYLING */
    .dark-panel {
        background-color: #101e4a; color: white; border-radius: 20px;
        padding: 25px; height: 100%; min-height: 500px;
        box-shadow: 0 10px 30px rgba(16, 30, 74, 0.5);
    }
    
    /* INPUT SIDEBAR STYLING */
    [data-testid="stSidebar"] { background-color: #ffffff; }
    
    h1, h2, h3, h4, h5 { font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING & LOGIC (From app (1).py)
# -----------------------------------------------------------------------------

# A fallback class in case 'reservoir_model.pkl' is not present
class MockModel:
    def predict(self, df):
        # A simple physics-inspired formula for fallback demonstration
        # Penman-Monteith simplified approximation
        t = df['Air_Temperature_C'].values[0]
        h = df['Humidity_pct'].values[0]
        w = df['Wind_Speed_kmh'].values[0]
        s = df['Solar_Radiation_kWh'].values[0]
        
        # Base evaporation + factors
        evap = (0.05 * t) + (0.03 * w) + (0.5 * s) - (0.02 * h)
        return [max(0.0, evap)] # Return as list to match sklearn format

@st.cache_resource
def load_model():
    try:
        model = joblib.load('reservoir_model.pkl')
        return model, True
    except FileNotFoundError:
        return MockModel(), False

model, is_real_model = load_model()

# -----------------------------------------------------------------------------
# 3. SIDEBAR INPUTS (From app (1).py)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/1200px-Flag_of_India.svg.png", width=50)
    st.title("⚙️ Parameters")
    st.caption("CWPRS AI Project | Shaneel S. Sao")
    
    st.divider()
    
    # Inputs
    temp = st.slider("Air Temperature (°C)", 0.0, 50.0, 30.0)
    humidity = st.slider("Humidity (%)", 0, 100, 40)
    wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0)
    solar = st.slider("Solar Radiation (kWh/m²)", 0.0, 12.0, 6.0)
    area = st.slider("Reservoir Surface Area (km²)", 0.1, 50.0, 5.0)
    
    st.info(f"**Current Inputs**\n\nTemp: {temp}°C\nWind: {wind} km/h")

    # Create DataFrame for Model
    input_df = pd.DataFrame({
        'Air_Temperature_C': [temp],
        'Humidity_pct': [humidity],
        'Wind_Speed_kmh': [wind],
        'Solar_Radiation_kWh': [solar],
        'Surface_Area_km2': [area]
    })

# -----------------------------------------------------------------------------
# 4. PERFORM PREDICTION
# -----------------------------------------------------------------------------
prediction_mm = model.predict(input_df)[0]

# Calculate Volumetric Loss
# 1 mm depth over 1 km^2 = 1,000,000 Liters (1 ML)
total_loss_liters = prediction_mm * (area * 1e6)
total_loss_ML = total_loss_liters / 1e6

# Determine status for UI
status = "High Evap" if prediction_mm > 6 else "Moderate" if prediction_mm > 3 else "Low Evap"
status_color = "#ff4b4b" if prediction_mm > 6 else "#ffa421" if prediction_mm > 3 else "#21c354"

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD UI
# -----------------------------------------------------------------------------
col_main, col_right = st.columns([2.5, 1])

with col_main:
    # --- HEADER CARD ---
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                padding: 30px; border-radius: 20px; color: white; margin-bottom: 20px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.15);">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h2 style="margin:0; font-weight: 300;">CWPRS Hydrology AI</h2>
                <h1 style="margin:10px 0; font-size: 3rem; font-weight: 700;">{datetime.now().strftime('%H:%M')}</h1>
                <p style="opacity: 0.8;">{datetime.now().strftime('%A, %B %d')} | Reservoir ID: 4B-Alpha</p>
            </div>
            <div style="text-align: right;">
                <h3 style="margin:0;">Status</h3>
                <p style="font-size: 1.5rem; font-weight:bold; color: {status_color}; text-shadow: 0px 0px 10px rgba(0,0,0,0.5);">{status}</p>
                <p style="opacity: 0.8;">Guided by: Dr. M Selva Balan</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_real_model:
        st.warning("⚠️ Using Physics Simulation (Mock Model). 'reservoir_model.pkl' not found.")

    # --- CHARTS (Simulated Visuals based on input intensity) ---
    with st.container():
        st.subheader("Hourly Evaporation Trend (Projected)")
        
        # Generate trend curve based on predicted max value
        hours = [f"{i}:00" for i in range(6, 19)]
        # Curve shape (bell curve) scaled by prediction
        curve = [0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        scaled_curve = [x * prediction_mm for x in curve]
        
        df_hourly = pd.DataFrame({"Hour": hours, "Evaporation (mm/hr)": scaled_curve})
        
        fig_hourly = px.line(df_hourly, x="Hour", y="Evaporation (mm/hr)", markers=True)
        fig_hourly.update_traces(line_color='#2a5298', line_shape='spline', fill='tozeroy', fillcolor='rgba(42, 82, 152, 0.1)')
        fig_hourly.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(t=20, b=20, l=20, r=20), height=250,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee')
        )
        st.plotly_chart(fig_hourly, use_container_width=True, config={'displayModeBar': False})

    # --- BOTTOM ROW ---
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("Volumetric Loss Impact")
        # Comparison Bar Chart
        categories = ['Current Loss', 'Avg Loss']
        values = [total_loss_ML, total_loss_ML * 0.8] # Dummy Avg
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Million Liters', x=categories, y=values, marker_color=['#1e3c72', '#aab7d1'])
        ])
        fig_bar.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(t=20, b=20, l=20, r=20), height=200,
            yaxis_title="Million Liters"
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with row2_col2:
        st.subheader("Regional Context")
        # Dummy Map for Visual Aesthetic
        df_map = pd.DataFrame({
            'lat': [18.5204, 19.0760, 18.50, 18.60], # Pune/Mumbai region coords
            'lon': [73.8567, 72.8777, 73.80, 73.90],
            'val': [prediction_mm, prediction_mm*0.8, prediction_mm*1.1, prediction_mm*0.9],
            'loc': ['CWPRS HQ', 'Mumbai', 'Khadakwasla', 'Panshet']
        })
        fig_map = px.scatter_geo(
            df_map, lat='lat', lon='lon', size='val', color='val',
            scope='asia', fitbounds="locations", color_continuous_scale='Blues'
        )
        fig_map.update_geos(visible=False, showcountries=True, countrycolor="lightgrey")
        fig_map.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})


# --- RIGHT COLUMN (DARK SIDEBAR - RESULTS) ---
with col_right:
    st.markdown(f"""
    <div class="dark-panel">
        <div style="text-align: center; margin-bottom: 30px;">
            <span style="font-size: 3rem;">☀️</span>
            <div style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 5px;">PREDICTED EVAPORATION</div>
            <h1 style="margin: 0; font-size: 3.5rem; color: #4facfe;">{prediction_mm:.2f}</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">mm / day</p>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; text-align: center; margin-bottom: 20px;">
            <small style="opacity:0.7">TOTAL VOLUME LOSS</small><br>
            <strong style="font-size: 1.8rem; color: #ff9a9e;">{total_loss_ML:.2f}</strong><br>
            <small>Million Liters</small>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 30px;">
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; text-align: center;">
                <small>Solar Rad</small><br>
                <strong>{solar} kWh</strong>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; text-align: center;">
                <small>Humidity</small><br>
                <strong>{humidity}%</strong>
            </div>
        </div>
        
        <h4 style="border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 10px;">Credits</h4>
        <div style="font-size: 0.8rem; opacity: 0.8;">
            <strong>Dev:</strong> Shaneel S. Sao<br>
            <strong>Guide:</strong> Smt. Anuja Rajgoplan<br>
            <strong>CWPRS AI Implimentation Cell</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
