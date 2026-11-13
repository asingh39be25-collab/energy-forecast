"""
Gradio Web App — Energy Consumption Forecaster
Trained on household_energy_weather_SYNTHETIC dataset
"""
import json, os, pickle
import gradio as gr
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".matplotlib")

from tensorflow.keras.models import load_model

MODEL_DIR = os.path.join(BASE_DIR, "model")

_model  = load_model(f"{MODEL_DIR}/lstm_energy.keras")
with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    _scaler = pickle.load(f)
with open(f"{MODEL_DIR}/feature_cols.json") as f:
    meta = json.load(f)

LOOKBACK      = meta["lookback"]
TARGET        = meta["target"]
FEATURE_COLS  = meta["feature_cols"]
ALL_COLS      = meta["all_cols"]


def predict_from_df(df: pd.DataFrame):
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        return None, f"❌ Missing columns: {missing}"
    df = df[ALL_COLS].dropna().iloc[-LOOKBACK:]
    if len(df) < LOOKBACK:
        return None, f"⚠️ Need at least {LOOKBACK} rows, got {len(df)}."
    scaled = _scaler.transform(df.values)
    X      = scaled[np.newaxis, ...]
    pred_s = _model.predict(X, verbose=0).flatten()[0]
    dummy  = np.zeros((1, len(ALL_COLS)))
    dummy[0, 0] = pred_s
    pred_kwh = _scaler.inverse_transform(dummy)[0, 0]
    return max(0.0, pred_kwh), None


def predict_csv(file_obj):
    if file_obj is None:
        return "⚠️ Please upload a CSV or Excel file."
    try:
        path = file_obj.name
        df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    except Exception as e:
        return f"❌ Could not read file: {e}"
    for col in ("timestamp", "Timestamp", "time", "Time"):
        if col in df.columns:
            df = df.drop(columns=[col])
    val, err = predict_from_df(df)
    if err:
        return err
    return f"### ⚡ Predicted next-hour energy: **{val:.4f} kWh**"


def predict_manual(temp, humidity, wind, precip, cloud, hour, dow, month, weekend,
                   *past_energy):
    rows = []
    for e in past_energy:
        rows.append({
            TARGET: e,
            "temperature_c": temp, "humidity_pct": humidity,
            "wind_speed_kmh": wind, "precipitation_mm": precip,
            "cloud_cover_pct": cloud, "hour": hour,
            "day_of_week": dow, "month": month, "is_weekend": weekend,
        })
    df = pd.DataFrame(rows)
    val, err = predict_from_df(df)
    if err:
        return err
    return f"### ⚡ Predicted next-hour energy: **{val:.4f} kWh**"


# ── UI ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="⚡ Energy Forecaster") as demo:
    gr.Markdown("""
    # ⚡ Household Energy Consumption Forecaster
    Predicts the **next hour's energy usage (kWh)** using an LSTM model.
    > Trained on 2 years of synthetic hourly data with weather features.
    """)

    with gr.Tab("📁 Upload CSV / Excel"):
        gr.Markdown(f"Upload a file with **at least {LOOKBACK} rows** and columns:\n"
                    f"`{', '.join(ALL_COLS)}`")
        file_in = gr.File(label="Upload .csv or .xlsx", file_types=[".csv", ".xlsx"])
        btn1    = gr.Button("🔮 Predict", variant="primary")
        out1    = gr.Markdown()
        btn1.click(predict_csv, inputs=file_in, outputs=out1)

    with gr.Tab("🖊️ Manual Entry"):
        gr.Markdown(f"Enter current weather + the last **{LOOKBACK} hours** of energy readings.")
        with gr.Row():
            temp    = gr.Number(label="Temperature (°C)",  value=22.0)
            humidity= gr.Number(label="Humidity (%)",       value=55)
            wind    = gr.Number(label="Wind Speed (km/h)",  value=8.0)
        with gr.Row():
            precip  = gr.Number(label="Precipitation (mm)", value=0.0)
            cloud   = gr.Number(label="Cloud Cover (%)",    value=30)
        with gr.Row():
            hour    = gr.Slider(0, 23, step=1, label="Hour of day",   value=12)
            dow     = gr.Slider(0, 6,  step=1, label="Day of week (0=Mon)", value=0)
            month   = gr.Slider(1, 12, step=1, label="Month",         value=6)
            weekend = gr.Radio([0, 1], label="Is Weekend?", value=0)

        gr.Markdown(f"**Last {LOOKBACK} hourly energy readings (kWh):** (oldest → newest)")
        past_inputs = [
            gr.Number(label=f"Hour -{LOOKBACK - i}", value=0.8)
            for i in range(LOOKBACK)
        ]
        btn2  = gr.Button("🔮 Predict", variant="primary")
        out2  = gr.Markdown()
        btn2.click(
            predict_manual,
            inputs=[temp, humidity, wind, precip, cloud, hour, dow, month, weekend] + past_inputs,
            outputs=out2,
        )

    gr.Markdown("---\n*LSTM model · Gradio · Synthetic dataset*")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

