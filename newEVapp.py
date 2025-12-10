import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Evaporation Intelligence", layout="wide")

# Custom CSS to mimic the UI design (Rounded corners, Dark/Light split, Shadows)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #eef2f7;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* CARD STYLING: White cards with shadows */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* CUSTOM METRIC CONTAINERS */
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }

    /* DARK SIDEBAR STYLING (Right Column mimic) */
    .dark-panel {
        background-color: #101e4a;
        color: white;
        border-radius: 20px;
        padding: 25px;
        height: 100%;
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DUMMY DATA GENERATION (Evaporation Specific)
# -----------------------------------------------------------------------------
def get_hourly_data():
    hours = [f"{i}:00 {'AM' if i < 12 else 'PM'}" for i in range(6, 19)] # 6am to 6pm
    # Evaporation usually peaks mid-day with temperature
    evap_rates = [0.2, 0.4, 0.8, 1.5, 2.2, 2.8, 3.1, 2.9, 2.4, 1.8, 1.1, 0.5, 0.3]
    return pd.DataFrame({"Hour": hours, "Evaporation (mm/hr)": evap_rates})

def get_monthly_data():
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Simulating High Evaporation in Summer, Low in Winter
    evap = [40, 45, 70, 95, 120, 110, 100, 90, 80, 60, 50, 42]
    rain = [20, 30, 10, 5, 10, 150, 200, 180, 120, 40, 15, 10]
    return pd.DataFrame({"Month": months, "Total Evaporation (mm)": evap, "Rainfall (mm)": rain})

def get_forecast_data():
    # Future predictions
    data = []
    base_date = datetime.now()
    for i in range(1, 5):
        day = base_date + timedelta(days=i)
        data.append({
            "Date": day.strftime("%A, %b %d"),
            "Condition": np.random.choice(["High Evap", "Moderate", "Low Evap"]),
            "Rate": f"{np.random.randint(4, 9)} mm/day"
        })
    return pd.DataFrame(data)

df_hourly = get_hourly_data()
df_monthly = get_monthly_data()
df_forecast = get_forecast_data()

# -----------------------------------------------------------------------------
# 3. LAYOUT STRUCTURE
# -----------------------------------------------------------------------------

# Split layout: 70% Left (Main Dashboard), 30% Right (Dark Panel)
col_main, col_right = st.columns([2.5, 1])

# --- LEFT COLUMN (MAIN CONTENT) ---
with col_main:
    # -- TOP HEADER CARD (HTML/CSS) --
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                padding: 30px; border-radius: 20px; color: white; margin-bottom: 20px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.15);">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h2 style="margin:0; font-weight: 300;">Good Morning, Engineer</h2>
                <h1 style="margin:10px 0; font-size: 3rem; font-weight: 700;">{datetime.now().strftime('%H:%M %p')}</h1>
                <p style="opacity: 0.8;">{datetime.now().strftime('%A, %B %d')} | Rajshahi, Bangladesh (Station 4B)</p>
            </div>
            <div style="text-align: right;">
                <h3 style="margin:0;">Evaporation Alert</h3>
                <p style="font-size: 1.2rem; font-weight:bold;">High Potential</p>
                <p style="opacity: 0.8;">Wind Speed: 20 km/h | Low Humidity</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -- HOURLY CHART --
    with st.container():
        st.subheader("Hourly Evaporation Rate Prediction (mm)")
        
        # Plotly Line Chart
        fig_hourly = px.line(df_hourly, x="Hour", y="Evaporation (mm/hr)", markers=True)
        fig_hourly.update_traces(line_color='#2a5298', line_shape='spline', fill='tozeroy', fillcolor='rgba(42, 82, 152, 0.1)')
        fig_hourly.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(t=20, b=20, l=20, r=20),
            height=250,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#eee')
        )
        st.plotly_chart(fig_hourly, use_container_width=True, config={'displayModeBar': False})

    # -- BOTTOM ROW (BAR CHART + MAP) --
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("Monthly Water Balance")
        # Multi-bar chart for Evap vs Rain
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=df_monthly['Month'], y=df_monthly['Total Evaporation (mm)'], name='Evaporation', marker_color='#1e3c72'))
        fig_bar.add_trace(go.Bar(x=df_monthly['Month'], y=df_monthly['Rainfall (mm)'], name='Rainfall', marker_color='#aab7d1'))
        
        fig_bar.update_layout(
            barmode='group',
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(t=20, b=20, l=20, r=20),
            height=200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(showgrid=True, gridcolor='#eee')
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with row2_col2:
        st.subheader("Regional Evaporation Heatmap")
        # Dummy Map Data
        df_map = pd.DataFrame({
            'lat': [24.36, 23.81, 22.35, 24.89],
            'lon': [88.62, 90.41, 91.78, 91.86],
            'evap': [8.5, 6.2, 5.1, 7.0],
            'city': ['Rajshahi', 'Dhaka', 'Chittagong', 'Sylhet']
        })
        
        fig_map = px.scatter_geo(
            df_map, lat='lat', lon='lon', size='evap', color='evap',
            scope='asia', color_continuous_scale='Blues',
            hover_name='city', size_max=30
        )
        fig_map.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="lightgrey")
        fig_map.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})


# --- RIGHT COLUMN (DARK SIDEBAR) ---
with col_right:
    # We use a container and markdown to simulate the dark blue panel background
    st.markdown("""
    <div class="dark-panel">
        <div style="text-align: center; margin-bottom: 30px;">
            <span style="font-size: 4rem;">☀️</span>
            <h1 style="margin: 0; font-size: 3.5rem;">7.2</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">mm / day</p>
            <h3 style="margin-top: 10px;">High Evaporation</h3>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 30px;">
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; text-align: center;">
                <small>Wind</small><br>
                <strong>20 km/h</strong>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; text-align: center;">
                <small>Humidity</small><br>
                <strong>15%</strong>
            </div>
        </div>
        
        <h4 style="border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 10px;">4-Day Forecast</h4>
    """, unsafe_allow_html=True)
    
    # Forecast List
    for index, row in df_forecast.iterrows():
        icon = "☀️" if "High" in row['Condition'] else "☁️"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); margin-bottom: 10px; padding: 15px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; color: white;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <div>
                    <div style="font-weight: bold; font-size: 0.9rem;">{row['Date']}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">{row['Condition']}</div>
                </div>
            </div>
            <div style="font-weight: bold;">{row['Rate']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
