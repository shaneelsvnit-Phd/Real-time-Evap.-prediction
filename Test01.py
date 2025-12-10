import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import requests
from datetime import datetime


csv_file = 'reservoir_history.csv'

# Generate 10 rows of dummy data
data = []
now = datetime.now()

for i in range(10):
    # Go back 'i' hours
    time_step = now - timedelta(hours=10-i)
    
    row = {
        'Timestamp': time_step.strftime("%Y-%m-%d %H:%M:%S"),
        'Air_Temperature_C': np.random.uniform(25, 35),
        'Humidity_pct': np.random.uniform(30, 60),
        'Wind_Speed_kmh': np.random.uniform(5, 15),
        'Solar_Radiation_kWh': np.random.uniform(4, 7),
        'Surface_Area_km2': 25.0,
        'Predicted_Evaporation_mm': np.random.uniform(3, 8) # Fake prediction
    }
    data.append(row)

df = pd.DataFrame(data)
df.to_csv(csv_file, index=False)

print(f"✅ Created '{csv_file}' with 10 rows of dummy data for testing.")

# --- 1. Configuration ---
# REPLACE THIS WITH YOUR ACTUAL API KEY
API_KEY = "c7d1876532760025967b708e94c55282" 

# Reservoir Location (Example: Khadakwasla Dam, Pune)
LAT = "18.438"
LON = "73.764"
MODEL_PATH = '/content/reservoir_model.pkl'
CSV_FILE = '/content/pune_climate_data.csv'

# API Endpoint
URL = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

def fetch_live_data():
    """Fetches real-time weather data."""
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()

        # Feature Engineering
        wind_kmh = data['wind']['speed'] * 3.6
        cloud_cover = data['clouds']['all']
        solar_est = 8.0 * (1 - (cloud_cover / 100))
        
        live_features = {
            'Air_Temperature_C': data['main']['temp'],
            'Humidity_pct': data['main']['humidity'],
            'Wind_Speed_kmh': wind_kmh,
            'Solar_Radiation_kWh': solar_est,
            'Surface_Area_km2': 25.0
        }
        return live_features

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_prediction(data_dict, prediction):
    """
    Saves the input data + prediction to CSV.
    Checks if file exists to determine if headers are needed.
    """
    # 1. Add Timestamp and Prediction to the data
    row = data_dict.copy()
    row['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row['Predicted_Evaporation_mm'] = prediction
    
    # 2. Reorder columns to put Timestamp first
    cols = ['Timestamp', 'Air_Temperature_C', 'Humidity_pct', 
            'Wind_Speed_kmh', 'Solar_Radiation_kWh', 
            'Surface_Area_km2', 'Predicted_Evaporation_mm']
    
    df_row = pd.DataFrame([row], columns=cols)
    
    # 3. Append to CSV
    # mode='a' means append. header=False prevents repeating headers.
    file_exists = os.path.isfile(CSV_FILE)
    df_row.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)
    print(f"   [Saved to {CSV_FILE}]")

def run_real_time_twin():
    if not os.path.exists(MODEL_PATH):
        print("CRITICAL ERROR: Model file not found.")
        return

    model = joblib.load(MODEL_PATH)
    print(f"--- Digital Twin Running ---")
    print(f"Logging data to: {CSV_FILE}\n")

    while True:
        features = fetch_live_data()
        
        if features:
            # Prepare for Model
            model_cols = ['Air_Temperature_C', 'Humidity_pct', 'Wind_Speed_kmh', 'Solar_Radiation_kWh', 'Surface_Area_km2']
            df_live = pd.DataFrame([features], columns=model_cols)
            
            # Predict
            pred_loss = model.predict(df_live)[0]
            
            # Print to Console
            now = datetime.now().strftime("%H:%M:%S")
            print(f"{now} | Temp: {features['Air_Temperature_C']}C | Pred: {pred_loss:.4f} mm", end="")
            
            # Save to File
            save_prediction(features, pred_loss)
        
        # Wait 10 seconds
        time.sleep(10)

if __name__ == "__main__":
    run_real_time_twin()

# --- Page Config ---
st.set_page_config(
    page_title="H2O AI: Evaporation Predictor", 
    page_icon="💧", 
    layout="wide"  # Changed to 'wide' to accommodate graphs better
)

# --- Header Section ---
st.title("💧 SHANEEL Reservoir Evaporation Predictor - CWPRS")
st.markdown("""
    **AI-Powered Hydrological Forecasting** By Shaneel S. Sao, R.A.(Engineering)  
    Under Guidance of Dr. M Selva Balan, Additional Director & Smt. Anuja Rajgoplan, 'Sci- C'  
    
    *Description: Uses a Random Forest Machine Learning model to estimate daily water loss based on meteorological parameters.* *Developed for CWPRS AI Project Implementation Cell.*
""")

st.divider()

# --- Model Loading Logic ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('reservoir_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'reservoir_model.pkl' not found. Please run the training script first.")
        return None

model = load_model()

# --- Tabs for Dual Functionality ---
tab1, tab2 = st.tabs(["🧮 Manual Calculator", "📡 Real-Time Monitor"])

# ==========================================
# TAB 1: MANUAL CALCULATOR (Your Original Code)
# ==========================================
with tab1:
    st.header("Scenario Analysis (Digital Twin Simulation)")
    
    col_input, col_pred = st.columns([1, 1])
    
    with col_input:
        st.subheader("⚙️ Input Parameters")
        temp = st.slider("Air Temperature (°C)", 0.0, 50.0, 30.0)
        humidity = st.slider("Humidity (%)", 0, 100, 40)
        wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0)
        solar = st.slider("Solar Radiation (kWh/m²)", 0.0, 12.0, 6.0)
        area = st.number_input("Reservoir Surface Area (km²)", min_value=0.1, value=5.0)

        input_data = {
            'Air_Temperature_C': temp,
            'Humidity_pct': humidity,
            'Wind_Speed_kmh': wind,
            'Solar_Radiation_kWh': solar,
            'Surface_Area_km2': area
        }
        input_df = pd.DataFrame(input_data, index=[0])
        
        st.write("Current Input Vector:")
        st.dataframe(input_df, hide_index=True)

    with col_pred:
        st.subheader("Prediction Results")
        if st.button("Calculate Loss", type="primary"):
            if model:
                # Predict
                prediction_mm = model.predict(input_df)[0]

                # Calculate Total Volume Loss
                # 1 mm * 1 km² = 1 Million Liters (ML)
                total_loss_ML = prediction_mm * input_df['Surface_Area_km2'][0]

                st.success(f"📉 Predicted Depth Loss: **{prediction_mm:.2f} mm**")
                st.info(f"🌊 Volumetric Loss: **{total_loss_ML:.2f} Million Liters**")
                
                # Visual Gauge (Simple Progress Bar representation)
                st.write("Severity Indicator:")
                severity = min(prediction_mm / 10.0, 1.0) # Assume 10mm is max extreme
                st.progress(severity)
                if prediction_mm > 7.5:
                    st.error("⚠️ CRITICAL EVAPORATION WARNING")
            else:
                st.warning("Model not loaded.")

# ==========================================
# TAB 2: REAL-TIME MONITOR (The Dashboard)
# ==========================================
with tab2:
    st.header("📡 Live Sensor Data & Forecast")
    
    CSV_FILE = 'reservoir_history.csv'
    
    # Auto-refresh logic button
    if st.button("🔄 Refresh Live Data"):
        st.rerun()

    if os.path.exists(CSV_FILE):
        # Read Data
        try:
            df_history = pd.read_csv(CSV_FILE)
            
            # Sort by latest
            df_history = df_history.iloc[::-1]  
            
            # Metrics Row
            if not df_history.empty:
                latest = df_history.iloc[0]
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latest Temp", f"{latest['Air_Temperature_C']} °C")
                m2.metric("Latest Humidity", f"{latest['Humidity_pct']}%")
                m3.metric("Latest Wind", f"{latest['Wind_Speed_kmh']:.1f} km/h")
                m4.metric("PREDICTED LOSS", f"{latest['Predicted_Evaporation_mm']:.2f} mm", delta_color="inverse")
            
            st.divider()

            # Interactive Chart
            st.subheader("🌊 Evaporation Trend (Real-Time)")
            
            # Prepare data for chart (Time as index)
            chart_data = df_history.copy()
            chart_data['Timestamp'] = pd.to_datetime(chart_data['Timestamp'])
            chart_data = chart_data.set_index('Timestamp')
            
            # Show the graph
            st.line_chart(chart_data[['Predicted_Evaporation_mm']], color="#007acc")
            
            # Show Raw Data (Expandable)
            with st.expander("View Raw Data Logs"):
                st.dataframe(df_history)
                
        except Exception as e:
            st.error(f"Error reading data file: {e}")
    else:
        st.warning("Waiting for data... Ensure the 'Real-Time Digital Twin' script is running in the background.")
        st.info(f"Looking for file: {os.path.abspath(CSV_FILE)}")

# --- Footer ---
st.markdown("---")
st.caption("SHANEEL AI PROJECT FOR CWPRS | Digital Twin V1.0")
