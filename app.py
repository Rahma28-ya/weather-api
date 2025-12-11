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
origins = ["*"]  # Bisa diganti dengan URL frontend temanmu
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ROOT =====
@app.get("/")
def home():
    return {"message": "Weather App API is running"}

# ===== DOWNLOAD MODEL =====
def download_file(url, filename):
    if not os.path.exists(filename):
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

download_file("https://huggingface.co/rahma28/weather-api/resolve/main/model.pkl", "model.pkl")
download_file("https://huggingface.co/rahma28/weather-api/resolve/main/scaler.pkl", "scaler.pkl")
download_file("https://huggingface.co/rahma28/weather-api/resolve/main/selector.pkl", "selector.pkl")

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

print("Scaler expects:", scaler.feature_names_in_)
print("Selector expects:", selector.feature_names_in_)

# ===== PREDICT API =====
@app.get("/predict")
def predict_api():
    try:
        # ===== Ambil data cuaca =====
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
            return {"error": "Hourly data missing", "response": response}

        hourly = response["hourly"]

        # ===== Build DataFrame lengkap =====
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

        # ===== Pastikan semua kolom ada & urutannya sesuai scaler =====
        for col in scaler.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[scaler.feature_names_in_]

        # ===== Transform & Select Features =====
        X_scaled = scaler.transform(df)
        selector_input = pd.DataFrame(X_scaled, columns=scaler.feature_names_in_)
        selector_input = selector_input[selector.feature_names_in_]  # hanya 15 kolom
        X_selected = selector.transform(selector_input.values)

        preds = model.predict(X_selected)
        avg_pred = float(np.mean(preds))

        return {"status": "success", "prediksi_rata_rata_suhu": avg_pred}

    except Exception as e:
        return {"error": str(e)}

# ===== FRONTEND GRADIO =====
def get_prediction():
    try:
        url = "http://localhost:8000/predict"  # ganti jika deploy
        r = requests.get(url)
        if r.status_code != 200:
            return f"Error: API mengembalikan status {r.status_code}"
        return r.json()
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="🌤 Weather Prediction App") as ui:
    gr.Markdown("""
        <h1 style='text-align:center; font-size:36px;'>🌤 Weather Prediction App</h1>
        <p style='text-align:center; font-size:18px; color:#555;'>
            Mengambil prediksi cuaca terbaru menggunakan Machine Learning dan API Open-Meteo.
        </p>
    """)
    with gr.Row():
        output = gr.Textbox(label="Hasil Prediksi Suhu", lines=4)
    btn = gr.Button("Ambil Prediksi Cuaca")
    btn.click(fn=get_prediction, outputs=output)

# ===== Mount Gradio ke FastAPI =====
app = gr.mount_gradio_app(app, ui, path="/app")