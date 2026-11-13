---
title: Energy Forecast App
emoji: ⚡
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: "3.50.2"
app_file: app_synthetic.py
pinned: false
license: mit
---

# ⚡ Household Energy Consumption Forecaster

An LSTM model that predicts the **next hour's household energy usage (kWh)**.

## Features
- 📁 Upload CSV/Excel with 24+ hourly rows → get instant prediction
- 🖊️ Manual entry mode for quick experiments
- Trained on 17,544 hourly rows (2 years of data)

## Model
- Architecture: LSTM(128) → LSTM(64) → Dense(32) → Dense(1)
- Test MAE: **0.0744 kWh** | Test RMSE: **0.0962 kWh**
