"""
Train LSTM on household_energy_weather_SYNTHETIC.xlsx
Target : energy_kwh
Features: temperature_c, humidity_pct, wind_speed_kmh, precipitation_mm,
          cloud_cover_pct, hour, day_of_week, month, is_weekend

Usage:
    python3 train_synthetic.py --xlsx /path/to/file.xlsx
Outputs:
    model/lstm_energy.keras
    model/scaler.pkl
    model/feature_cols.json
"""

import argparse, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

LOOKBACK   = 24          # hours of history
TARGET     = "energy_kwh"
MODEL_DIR  = "model"
FEATURE_COLS = [
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "precipitation_mm", "cloud_cover_pct",
    "hour", "day_of_week", "month", "is_weekend"
]
ALL_COLS = [TARGET] + FEATURE_COLS   # target always first (index 0)


def load_data(xlsx_path):
    print(f"📂  Reading {xlsx_path} …")
    df = pd.read_excel(xlsx_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[ALL_COLS].dropna()
    print(f"    {len(df):,} rows, {len(df.columns)} columns — no nulls ✅")
    return df


def scale_and_sequence(df, lookback=LOOKBACK):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[ALL_COLS].values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback: i])
        y.append(scaled[i, 0])          # index 0 = target

    return np.array(X), np.array(y), scaler


def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def inverse_target(scaler, arr, n_feats):
    dummy = np.zeros((len(arr), n_feats))
    dummy[:, 0] = arr
    return scaler.inverse_transform(dummy)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx",   required=True, help="Path to .xlsx file")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(args.xlsx)
    X, y, scaler = scale_and_sequence(df)

    split = int(len(X) * (1 - args.test_frac))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    print(f"\n🔀  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    model = build_lstm((LOOKBACK, X.shape[2]))
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f"{MODEL_DIR}/lstm_energy.keras",
                        save_best_only=True, verbose=0),
    ]

    print(f"\n🏋️  Training for up to {args.epochs} epochs …")
    history = model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    n_feats = len(ALL_COLS)
    y_pred_s = model.predict(X_te, verbose=0).flatten()
    y_pred   = inverse_target(scaler, y_pred_s, n_feats)
    y_true   = inverse_target(scaler, y_te,     n_feats)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true,  y_pred) ** 0.5
    print(f"\n✅  Test MAE : {mae:.4f} kWh")
    print(f"   Test RMSE: {rmse:.4f} kWh")

    best_val = min(history.history["val_loss"])
    print(f"   Best val_loss: {best_val:.6f}")

    # ── Save artefacts ────────────────────────────────────────────────────
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_cols":  FEATURE_COLS,
        "all_cols":      ALL_COLS,
        "target":        TARGET,
        "lookback":      LOOKBACK,
        "mae":           round(float(mae),  5),
        "rmse":          round(float(rmse), 5),
    }
    with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾  Artefacts saved to ./{MODEL_DIR}/")
    print("    lstm_energy.keras  |  scaler.pkl  |  feature_cols.json")
    print("\n🚀  Run the app:  python3 app_synthetic.py")


if __name__ == "__main__":
    main()

