"""
Utility: prepare any CSV for inference.
Resamples minute/second data to hourly if needed.
Usage: python prepare_input.py <input.csv> [output.csv]
"""
import sys, pandas as pd

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "input_ready.csv"

df = pd.read_csv(src, sep=None, engine="python", na_values=["?", ""])

# detect timestamp column
for col in ("timestamp", "Timestamp", "datetime", "Date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        df = df.set_index(col)
        break
elif "Date" in df.columns and "Time" in df.columns:
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date","Time"]).set_index("timestamp")

df = df.apply(pd.to_numeric, errors="coerce")

# resample to hourly if needed (frequency > 1 min suggests already hourly)
freq = df.index.to_series().diff().median()
if freq and freq.total_seconds() < 3600:
    print(f"Resampling from {freq} to 1h …")
    df = df.resample("h").mean().interpolate(limit_direction="both")

df.to_csv(dst)
print(f"Saved {len(df)} hourly rows → {dst}")

