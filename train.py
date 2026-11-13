"""
Energy Consumption Forecasting — LSTM Model
============================================
Trains an LSTM to predict the next hour's Global_active_power
from the last N hours of features.

Usage:
    pip install -r requirements.txt
    python train.py --csv path/to/merged_hourly.csv

Outputs:
    model/scaler.pkl
    model/lstm_energy.h5   (or .keras)
    model/feature_cols.json
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── TensorFlow / Keras ─────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

LOOKBACK = 24          # hours of history fed to LSTM
FORECAST_HORIZON = 1   # predict 1 hour ahead
TARGET = "Global_active_power"
MODEL_DIR = "model"


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != TARGET]
    all_cols = [TARGET] + feature_cols          # target first for easy slicing

    data = df[all_cols].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler, feature_cols, df.index


def make_sequences(data_scaled, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(data_scaled) - FORECAST_HORIZON + 1):
        X.append(data_scaled[i - lookback: i])          # shape (lookback, n_features)
        y.append(data_scaled[i, 0])                     # target column index = 0
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to merged_hourly.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("📂  Loading data …")
    data_scaled, scaler, feature_cols, idx = load_and_prepare(args.csv)

    print("🔧  Building sequences …")
    X, y = make_sequences(data_scaled)

    split = int(len(X) * (1 - args.test_frac))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    model = build_model(input_shape=(LOOKBACK, X.shape[2]))
    model.summary()

    cb = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f"{MODEL_DIR}/lstm_energy.keras", save_best_only=True, verbose=0),
    ]

    print("\n🏋️  Training …")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=1,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test).flatten()

    # inverse-transform only the target column
    n_feats = data_scaled.shape[1]
    dummy = np.zeros((len(y_pred_scaled), n_feats))
    dummy[:, 0] = y_pred_scaled
    y_pred = scaler.inverse_transform(dummy)[:, 0]

    dummy2 = np.zeros((len(y_test), n_feats))
    dummy2[:, 0] = y_test
    y_true = scaler.inverse_transform(dummy2)[:, 0]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"\n✅  Test MAE : {mae:.4f} kW")
    print(f"   Test RMSE: {rmse:.4f} kW")

    # ── Save artefacts ────────────────────────────────────────────────────
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "lookback": LOOKBACK}, f)

    print(f"\n💾  Saved model artefacts to ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()

