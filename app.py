import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained models & scaler (must be saved beforehand)
models = pickle.load(open("models.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Rainfall Prediction", page_icon="🌧️", layout="wide")

st.title("🌦️ Rainfall Prediction Web App")
st.markdown("Enter weather parameters to predict **probability of rainfall**.")

# Sidebar inputs
st.sidebar.header("Weather Inputs")

pressure = st.sidebar.slider("Pressure (hPa)", 950, 1050, 1012)
temperature = st.sidebar.slider("Temperature (°C)", -10, 50, 28)
dewpoint = st.sidebar.slider("Dewpoint (°C)", -10, 40, 24)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 75)
cloud = st.sidebar.slider("Cloud Cover (oktas)", 0, 8, 6)
sunshine = st.sidebar.slider("Sunshine (hours)", 0, 24, 8)
winddirection = st.sidebar.slider("Wind Direction (degrees)", 0, 360, 180)
windspeed = st.sidebar.slider("Wind Speed (m/s)", 0, 50, 15)

# Arrange input
new_data = [[pressure, temperature, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]]
new_data_scaled = scaler.transform(new_data)

if st.button("🔍 Predict"):
    results = []
    for model in models:
        prob = model.predict_proba(new_data_scaled)[0][1]
        pred = model.predict(new_data_scaled)[0]
        results.append({
            "Model": model.__class__.__name__,
            "Rain Probability": prob,
            "Prediction": "🌧️ Rain" if pred == 1 else "☀️ No Rain"
        })
    
    df_results = pd.DataFrame(results)

    st.subheader("📊 Prediction Results")
    st.dataframe(df_results, hide_index=True)

    # Bar chart for probabilities
    st.subheader("🔢 Probability Chart")
    st.bar_chart(df_results.set_index("Model")["Rain Probability"])

models = pickle.load(open("models.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
