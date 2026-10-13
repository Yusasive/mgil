"""
MGIL Training Script — Mini-Grid Intelligence Layer
====================================================
Run this script ONCE during installation to train and save the models.

Usage:
    python mgil_train.py                    # Train on synthetic baseline data
    python mgil_train.py --data site.csv    # Train on real site data (recommended)

Output:
    models/lstm_model.keras
    models/rf_model.pkl
    models/solar_rf_model.pkl   ← NEW: ML solar generation forecast model
    models/scaler.pkl
    models/training_report.txt
"""

import argparse
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time and seasonal features to a DataFrame with a DatetimeIndex."""
    df = df.copy()
    df["hour"]        = df.index.hour
    df["dayofweek"]   = df.index.dayofweek
    df["month"]       = df.index.month

    # Nigerian seasonal flags
    df["is_harmattan"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["is_harvest"]   = df["month"].isin([9, 10, 11]).astype(int)
    # Market day: every 4th day from 1 Jan
    df["is_market_day"] = (
        (df.index.dayofyear % 4 == 0).astype(int)
    )

    # Lag features (hours back)
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}h"] = df["load_kw"].shift(lag)

    df.dropna(inplace=True)
    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for solar generation forecasting.
    Solar is driven by time-of-day, season, and lag irradiance patterns.
    """
    df = df.copy()
    df["hour"]           = df.index.hour
    df["month"]          = df.index.month
    df["dayofyear"]      = df.index.dayofyear
    df["is_harmattan"]   = df["month"].isin([11, 12, 1, 2]).astype(int)

    # Solar angle proxy: sin of hour rescaled to daytime window
    # Peaks at solar noon (~13:00 Nigerian Standard Time)
    df["solar_angle"]    = np.sin(np.pi * (df["hour"] - 6) / 12).clip(0)

    # Day length varies with season (proxy via dayofyear cosine)
    df["day_length"]     = 1.0 + 0.1 * np.cos(2 * np.pi * (df["dayofyear"] - 172) / 365)

    # Lag solar readings (detect cloud patterns)
    for lag in [1, 2, 3]:
        df[f"solar_lag_{lag}h"] = df["solar_kw"].shift(lag)

    df.dropna(inplace=True)
    return df


def build_sequences(series: np.ndarray, seq_len: int = 24):
    """Convert a 1-D array into (X, y) sequences for LSTM."""
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_real_data(path: str) -> pd.DataFrame:
    """
    Load real site CSV.  Expected columns (flexible):
        datetime / timestamp / date → parsed as index
        load_kw  / load / demand    → renamed to load_kw
        solar_kw / pv / irradiance  → renamed to solar_kw  (optional)
        bess_soc / soc              → renamed to bess_soc  (optional)
    """
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.lower().str.strip()
    time_cols = [c for c in df.columns if c in ("datetime", "timestamp", "date", "time")]
    if time_cols:
        df.index = pd.to_datetime(df[time_cols[0]])
        df.drop(columns=time_cols, inplace=True)
    else:
        df.index = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]

    rename = {}
    for c in df.columns:
        if c in ("load", "demand", "load_kw"):
            rename[c] = "load_kw"
        elif c in ("solar", "pv", "irradiance", "solar_kw"):
            rename[c] = "solar_kw"
        elif c in ("soc", "bess_soc"):
            rename[c] = "bess_soc"
    df.rename(columns=rename, inplace=True)

    if "load_kw" not in df.columns:
        raise ValueError("CSV must contain a load column (load_kw, load, or demand).")

    df = df.resample("1h").mean().interpolate()
    return df


def generate_synthetic_data(n_days: int = 365) -> pd.DataFrame:
    """
    Simulate one year of hourly mini-grid data for a 100-kW DARES site.
    This is only used when no real data is provided.
    """
    hours = n_days * 24
    idx = pd.date_range("2024-01-01", periods=hours, freq="1h")

    hour        = idx.hour.values
    month       = idx.month.values
    is_harmattan = np.isin(month, [11, 12, 1, 2]).astype(float)
    is_harvest   = np.isin(month, [9, 10, 11]).astype(float)

    # Load profile: morning + evening peaks
    base = 25.0
    load = (
        base
        + 15 * np.sin(np.pi * (hour - 6) / 12) * (hour >= 6) * (hour <= 18)
        + 20 * np.exp(-((hour - 19) ** 2) / 4)
        + 8 * is_harvest
        - 5 * is_harmattan
        + np.random.normal(0, 3, hours)
    ).clip(5)

    # Solar: daytime Gaussian, reduced in Harmattan haze
    solar = (
        55 * np.exp(-((hour - 13) ** 2) / 10) * (hour >= 6) * (hour <= 18)
        * (1 - 0.25 * is_harmattan)
        + np.random.normal(0, 2, hours)
    ).clip(0)

    # BESS SoC: simple daily cycle
    soc = 50 + 30 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5, hours)
    soc = np.clip(soc, 20, 95)

    return pd.DataFrame({"load_kw": load, "solar_kw": solar, "bess_soc": soc}, index=idx)


# ── Model builders ────────────────────────────────────────────────────────────
def build_lstm(seq_len: int = 24) -> Sequential:
    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(data_path: str | None = None):
    print("=" * 60)
    print("  MGIL Training — Mini-Grid Intelligence Layer")
    print("=" * 60)

    # 1. Load data
    if data_path:
        print(f"\n[1/5] Loading real site data: {data_path}")
        df = load_real_data(data_path)
        print(f"      {len(df):,} hourly records loaded.")
    else:
        print("\n[1/5] No site data provided — generating synthetic baseline.")
        df = generate_synthetic_data(365)
        print("      365-day synthetic dataset created (100 kW DARES profile).")

    # Ensure solar / soc columns exist (may be absent in real data)
    if "solar_kw" not in df.columns:
        df["solar_kw"] = 0.0
    if "bess_soc" not in df.columns:
        df["bess_soc"] = 50.0

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    df = add_features(df)

    FEATURE_COLS = [
        "hour", "dayofweek", "month",
        "is_market_day", "is_harmattan", "is_harvest",
        "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_12h", "lag_24h",
    ]
    TARGET = "load_kw"

    # Train / test split (80 / 20, time-ordered)
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"      Train: {len(train_df):,} h  |  Test: {len(test_df):,} h")

    # 3. Random Forest (load)
    print("\n[3/6] Training Load Random Forest (200 trees)...")
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET].values

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_mape  = mean_absolute_percentage_error(y_test, rf_pred) * 100
    print(f"      RF MAPE on test set: {rf_mape:.2f}%")

    # 4. Solar RF
    print("\n[4/6] Training Solar Generation Random Forest (100 trees)...")
    SOLAR_FEATURE_COLS = [
        "hour", "month", "dayofyear", "is_harmattan",
        "solar_angle", "day_length",
        "solar_lag_1h", "solar_lag_2h", "solar_lag_3h",
    ]
    df_solar = add_solar_features(df[["load_kw", "solar_kw"]].copy())
    solar_split = int(len(df_solar) * 0.8)
    solar_train = df_solar.iloc[:solar_split]
    solar_test  = df_solar.iloc[solar_split:]

    Xs_train = solar_train[SOLAR_FEATURE_COLS].values
    ys_train = solar_train["solar_kw"].values
    Xs_test  = solar_test[SOLAR_FEATURE_COLS].values
    ys_test  = solar_test["solar_kw"].values

    solar_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    solar_rf.fit(Xs_train, ys_train)
    solar_pred = solar_rf.predict(Xs_test)

    # Clip negatives — solar can't be negative
    solar_pred  = np.clip(solar_pred, 0, None)
    # MAPE only on daytime hours (avoid division by near-zero at night)
    daytime_mask = ys_test > 1.0
    if daytime_mask.sum() > 0:
        solar_mape = mean_absolute_percentage_error(
            ys_test[daytime_mask], solar_pred[daytime_mask]
        ) * 100
    else:
        solar_mape = 0.0
    print(f"      Solar RF MAPE (daytime hours): {solar_mape:.2f}%")

    # 5. LSTM
    print("\n[5/6] Training LSTM (2-layer, 64+32 units)...")
    scaler = MinMaxScaler()
    load_scaled = scaler.fit_transform(df[[TARGET]]).flatten()

    SEQ_LEN = 24
    split_scaled = int(len(load_scaled) * 0.8)
    X_lstm_train, y_lstm_train = build_sequences(load_scaled[:split_scaled], SEQ_LEN)
    X_lstm_test,  y_lstm_test  = build_sequences(load_scaled[split_scaled:], SEQ_LEN)

    X_lstm_train = X_lstm_train[..., np.newaxis]
    X_lstm_test  = X_lstm_test[..., np.newaxis]

    lstm_model = build_lstm(SEQ_LEN)
    ckpt_path  = os.path.join(MODELS_DIR, "lstm_best.keras")

    lstm_model.fit(
        X_lstm_train, y_lstm_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(ckpt_path, save_best_only=True, verbose=0),
        ],
        verbose=1,
    )

    lstm_pred_scaled = lstm_model.predict(X_lstm_test, verbose=0).flatten()
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
    lstm_actual = scaler.inverse_transform(y_lstm_test.reshape(-1, 1)).flatten()
    lstm_mape = mean_absolute_percentage_error(lstm_actual, lstm_pred) * 100
    print(f"      LSTM MAPE on test set: {lstm_mape:.2f}%")

    # 6. Save models
    print("\n[6/6] Saving models...")
    lstm_path  = os.path.join(MODELS_DIR, "lstm_model.keras")
    rf_path    = os.path.join(MODELS_DIR, "rf_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    solar_rf_path = os.path.join(MODELS_DIR, "solar_rf_model.pkl")

    lstm_model.save(lstm_path)
    joblib.dump(rf, rf_path)
    joblib.dump(solar_rf, solar_rf_path)
    joblib.dump(scaler, scaler_path)

    # Save metadata
    meta = {
        "trained_on": pd.Timestamp.now().isoformat(),
        "data_source": data_path or "synthetic",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "rf_mape": round(rf_mape, 4),
        "lstm_mape": round(lstm_mape, 4),
        "solar_rf_mape": round(solar_mape, 4),
        "feature_cols": FEATURE_COLS,
        "solar_feature_cols": SOLAR_FEATURE_COLS,
        "seq_len": SEQ_LEN,
    }
    meta_path = os.path.join(MODELS_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Training report
    report = f"""MGIL Training Report
====================
Date:           {meta['trained_on']}
Data source:    {meta['data_source']}
Train samples:  {meta['train_samples']:,} hours
Test samples:   {meta['test_samples']:,} hours

Model Performance
-----------------
Load Random Forest MAPE:   {rf_mape:.2f}%
Load LSTM MAPE:            {lstm_mape:.2f}%
Solar RF MAPE (daytime):   {solar_mape:.2f}%

Saved Files
-----------
{lstm_path}
{rf_path}
{solar_rf_path}
{scaler_path}
{meta_path}
"""
    report_path = os.path.join(MODELS_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n  ✓  lstm_model.keras")
    print(f"  ✓  rf_model.pkl")
    print(f"  ✓  solar_rf_model.pkl")
    print(f"  ✓  scaler.pkl")
    print(f"  ✓  meta.json")
    print(f"  ✓  training_report.txt")
    print(f"\n  Load RF MAPE:         {rf_mape:.2f}%")
    print(f"  Load LSTM MAPE:       {lstm_mape:.2f}%")
    print(f"  Solar RF MAPE:        {solar_mape:.2f}%")
    print("\n  Training complete. Run mgil_predict.py to forecast.\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MGIL models")
    parser.add_argument(
        "--data", metavar="CSV",
        help="Path to site CSV file (optional; uses synthetic data if omitted)"
    )
    args = parser.parse_args()
    train(args.data)