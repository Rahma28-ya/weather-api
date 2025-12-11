import numpy as np
import pandas as pd
import joblib
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os

app = FastAPI()

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bisa diganti URL frontend temanmu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ROOT =====
@app.get("/")
def home():
    return {"message": "Weather App API is running"}

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# ===== PREDICT API =====
@app.get("/predict")
def predict_api():
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": -4.087,
            "longitude": 137.123,
            "hourly": "temperature_2m,relativehumidity_2m,dewpoint_2m,rain,pressure_msl,"
                      "cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,"
                      "windspeed_10m,winddirection_10m",
            "forecast_days": 1,
            "timezone": "Asia/Singapore"
        }

        response = requests.get(url, params=params).json()
        if "hourly" not in response:
            return {"error": "Hourly data missing"}

        hourly = response["hourly"]

        data_dict = {
            "temperature": hourly.get("temperature_2m", [0]*24),
            "relative_humidity": hourly.get("relativehumidity_2m", [0]*24),
            "dew_point": hourly.get("dewpoint_2m", [0]*24),
            "rain (mm)": hourly.get("rain", [0]*24),
            "pressure_msl (hPa)": hourly.get("pressure_msl", [1013]*24),
            "cloud_cover (%)": hourly.get("cloudcover", [0]*24),
            "cloud_cover_low (%)": hourly.get("cloudcover_low", [0]*24),
            "cloud_cover_mid (%)": hourly.get("cloudcover_mid", [0]*24),
            "cloud_cover_high (%)": hourly.get("cloudcover_high", [0]*24),
            "wind_speed_10m (km/h)": hourly.get("windspeed_10m", [0]*24),
            "wind_direction": hourly.get("winddirection_10m", [0]*24),
            "is_Day": [1]*24,
            "precipitation (mm)": [0]*24,
            "snowfall (cm)": [0]*24,
            "surface_pressure (hPa)": hourly.get("pressure_msl", [1013]*24),
            "vapour_pressure_deficit (kPa)": [0]*24,
        }

        df = pd.DataFrame(data_dict)

        # Pastikan urutan kolom sesuai scaler
        for col in scaler.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[scaler.feature_names_in_]

        X_scaled = scaler.transform(df)
        selector_input = pd.DataFrame(X_scaled, columns=scaler.feature_names_in_)
        selector_input = selector_input[selector.feature_names_in_]
        X_selected = selector.transform(selector_input.values)

        preds = model.predict(X_selected)
avg_pred = float(np.mean(preds))

return {
    "status": "success",
    "prediksi_rata_rata_suhu": avg_pred,
    "sample_data": df.head(3).to_dict(orient="records")
}
    except Exception as e:
        return {"error": str(e)}

# ===== GRADIO FRONTEND =====
def get_prediction():
    r = requests.get("http://localhost:8000/predict")  # Railway nanti ganti domain
    if r.status_code == 200:
        return str(r.json())
    return f"Error: {r.status_code}"

with gr.Blocks() as ui:
    gr.Markdown("<h1 style='text-align:center;'>🌤 Weather Prediction App</h1>")
    output = gr.Textbox(label="Hasil Prediksi Suhu", lines=4)
    btn = gr.Button("Ambil Prediksi Cuaca")
    btn.click(fn=get_prediction, outputs=output)

# Mount Gradio ke FastAPI
app = gr.mount_gradio_app(app, ui, path="/app")


