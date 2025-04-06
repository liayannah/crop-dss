import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np

# --- TRAINING DATA ---
crop_data = pd.DataFrame({
    'Yield': [3000, 3200, 2900, 3100, 3300],
    'Fertilizer': [100, 120, 90, 110, 130],
    'Irrigation': [5000, 5500, 4500, 5200, 5700],
    'Labor': [200, 220, 190, 210, 230],
    'SoilQuality': [7, 8, 6.5, 7.5, 8.5],
    'WeatherIndex': [70, 75, 65, 72, 78]
})

pest_data = pd.DataFrame({
    'PestIncidence': [15, 20, 10, 18, 25],
    'Temperature': [28, 30, 27, 29, 31],
    'Humidity': [60, 65, 55, 62, 68],
    'CropStage': [1, 2, 1, 2, 3],
    'Pesticide': [1, 0, 1, 0, 1],
    'TimeSinceSpray': [5, 10, 3, 7, 2]
})

# --- MODEL TRAINING ---
X_yield = sm.add_constant(crop_data[['Fertilizer', 'Irrigation', 'Labor', 'SoilQuality', 'WeatherIndex']])
y_yield = crop_data['Yield']
model_yield = sm.OLS(y_yield, X_yield).fit()

X_pest = sm.add_constant(pest_data[['Temperature', 'Humidity', 'CropStage', 'Pesticide', 'TimeSinceSpray']])
y_pest = pest_data['PestIncidence']
model_pest = sm.OLS(y_pest, X_pest).fit()

# --- STREAMLIT APP ---
st.title("Crop Management & Pest Control DSS")

st.header("1. Input for Crop Resource Allocation")
fert = st.slider("Fertilizer (kg/ha)", 80, 150, 110)
irrig = st.slider("Irrigation (liters/ha)", 4000, 6000, 5200)
labor = st.slider("Labor Hours", 150, 250, 210)
soil = st.slider("Soil Quality Index", 5.0, 9.0, 7.5)
weather = st.slider("Weather Index", 60, 80, 72)

input_yield = sm.add_constant(pd.DataFrame([{
    'Fertilizer': fert,
    'Irrigation': irrig,
    'Labor': labor,
    'SoilQuality': soil,
    'WeatherIndex': weather
}]))
yield_prediction = model_yield.predict(input_yield)[0]
st.success(f"Predicted Crop Yield: {yield_prediction:.2f} kg/ha")

st.header("2. Input for Pest Control Decision")
temp = st.slider("Temperature (°C)", 25, 35, 29)
humid = st.slider("Humidity (%)", 50, 80, 64)
stage = st.selectbox("Crop Stage", {'1 - Vegetative': 1, '2 - Flowering': 2, '3 - Fruiting': 3})
pesticide = st.radio("Was pesticide recently applied?", ["Yes", "No"])
time_since_spray = st.slider("Days Since Last Spray", 0, 15, 6)

input_pest = sm.add_constant(pd.DataFrame([{
    'Temperature': temp,
    'Humidity': humid,
    'CropStage': int(stage[-1]),
    'Pesticide': 1 if pesticide == "Yes" else 0,
    'TimeSinceSpray': time_since_spray
}]))
pest_prediction = model_pest.predict(input_pest)[0]
st.warning(f"Predicted Pest Incidence: {pest_prediction:.2f} pests/m²")

st.caption("Prototype DSS using regression analysis for agriculture optimization.")
