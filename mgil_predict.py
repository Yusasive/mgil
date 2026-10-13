"""
MGIL Prediction Script — Mini-Grid Intelligence Layer
=====================================================
Run this script daily (via cron) to get a 24-hour load forecast
and SEMS dispatch decisions for each hour.

Usage:
    python mgil_predict.py                          # Uses current time + last 24h from log
    python mgil_predict.py --input readings.csv     # Load last 24h from a CSV file
    python mgil_predict.py --soc 72.5               # Override current BESS SoC reading

Cron example (runs every day at 06:00):
    0 6 * * * /home/pi/mgil/venv/bin/python /home/pi/mgil/mgil_predict.py >> /home/pi/mgil/logs/mgil.log 2>&1

Output:
    forecast/forecast_YYYY-MM-DD.csv   — 24-hour load forecast
    forecast/dispatch_YYYY-MM-DD.csv   — SEMS dispatch decisions
    forecast/report_YYYY-MM-DD.txt     — human-readable summary
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
FORECAST_DIR = os.path.join(os.path.dirname(__file__), "forecast")
LOG_DIR      = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)

# ── SEMS parameters (match paper defaults) ───────────────────────────────────
SEMS_PARAMS = {
    "grid_capacity_kw":   100.0,
    "pv_capacity_kw":      75.0,
    "bess_capacity_kwh":  200.0,
    "diesel_capacity_kw":  50.0,
    "bess_min_soc":        20.0,   # %
    "bess_max_soc":        95.0,   # %
    "bess_charge_rate":    40.0,   # kW
    "bess_discharge_rate": 50.0,   # kW
    "diesel_min_load_pct": 30.0,   # % of rated capacity
    "lookahead_hours":      6,
}


# ── Feature helpers ───────────────────────────────────────────────────────────
def make_time_features(dt: pd.Timestamp) -> dict:
    return {
        "hour":          dt.hour,
        "dayofweek":     dt.dayofweek,
        "month":         dt.month,
        "is_harmattan":  int(dt.month in (11, 12, 1, 2)),
        "is_harvest":    int(dt.month in (9, 10, 11)),
        "is_market_day": int(dt.dayofyear % 4 == 0),
    }


def make_rf_features(history: np.ndarray, dt: pd.Timestamp) -> np.ndarray:
    """Build the same 12-feature vector used at training time."""
    tf_feats = make_time_features(dt)
    lags = {}
    for lag in [1, 2, 3, 6, 12, 24]:
        idx = -lag
        lags[f"lag_{lag}h"] = history[idx] if len(history) >= lag else history[0]

    row = [
        tf_feats["hour"],
        tf_feats["dayofweek"],
        tf_feats["month"],
        tf_feats["is_market_day"],
        tf_feats["is_harmattan"],
        tf_feats["is_harvest"],
        lags["lag_1h"],
        lags["lag_2h"],
        lags["lag_3h"],
        lags["lag_6h"],
        lags["lag_12h"],
        lags["lag_24h"],
    ]
    return np.array(row).reshape(1, -1)


# ── SEMS dispatch logic ───────────────────────────────────────────────────────
def sems_dispatch(
    load_kw:    float,
    solar_kw:   float,
    bess_soc:   float,      # current SoC in %
    hour:       int,
    p:          dict = SEMS_PARAMS,
) -> dict:
    """
    Rule-based dispatch for one hour.
    Returns state, power flows, and updated SoC.
    """
    net = solar_kw - load_kw          # positive → surplus, negative → deficit
    bess_kwh_avail = (bess_soc - p["bess_min_soc"]) / 100 * p["bess_capacity_kwh"]
    bess_kwh_room  = (p["bess_max_soc"] - bess_soc) / 100 * p["bess_capacity_kwh"]

    dispatch = {
        "load_kw":       round(load_kw, 2),
        "solar_kw":      round(solar_kw, 2),
        "bess_soc_start": round(bess_soc, 1),
        "pv_to_load":    0.0,
        "pv_to_bess":    0.0,
        "bess_to_load":  0.0,
        "diesel_kw":     0.0,
        "curtailed_kw":  0.0,
        "state":         "",
        "bess_soc_end":  0.0,
    }

    if net >= 0:
        # ── PV Surplus ──────────────────────────────────────────────────────
        dispatch["state"]       = "PV_SURPLUS"
        dispatch["pv_to_load"]  = round(load_kw, 2)
        charge                  = min(net, p["bess_charge_rate"], bess_kwh_room)
        dispatch["pv_to_bess"]  = round(charge, 2)
        dispatch["curtailed_kw"]= round(net - charge, 2)
        new_soc = bess_soc + (charge / p["bess_capacity_kwh"]) * 100

    elif bess_kwh_avail >= abs(net):
        # ── BESS covers deficit ─────────────────────────────────────────────
        dispatch["state"]       = "BESS_COVER"
        dispatch["pv_to_load"]  = round(solar_kw, 2)
        bess_discharge          = min(abs(net), p["bess_discharge_rate"])
        dispatch["bess_to_load"] = round(bess_discharge, 2)
        new_soc = bess_soc - (bess_discharge / p["bess_capacity_kwh"]) * 100

    elif solar_kw > 0 and bess_kwh_avail > 0:
        # ── Partial cover, diesel required ──────────────────────────────────
        dispatch["state"]       = "DIESEL_REQUIRED"
        dispatch["pv_to_load"]  = round(solar_kw, 2)
        bess_discharge          = min(bess_kwh_avail, p["bess_discharge_rate"])
        dispatch["bess_to_load"] = round(bess_discharge, 2)
        unmet                   = load_kw - solar_kw - bess_discharge
        diesel                  = max(unmet, p["diesel_capacity_kw"] * p["diesel_min_load_pct"] / 100)
        dispatch["diesel_kw"]   = round(min(diesel, p["diesel_capacity_kw"]), 2)
        new_soc = bess_soc - (bess_discharge / p["bess_capacity_kwh"]) * 100

    else:
        # ── Night / full diesel ─────────────────────────────────────────────
        dispatch["state"]       = "DIESEL_REQUIRED"
        dispatch["pv_to_load"]  = 0.0
        bess_discharge          = min(bess_kwh_avail, p["bess_discharge_rate"], load_kw)
        dispatch["bess_to_load"] = round(bess_discharge, 2)
        unmet                   = load_kw - bess_discharge
        diesel                  = max(unmet, p["diesel_capacity_kw"] * p["diesel_min_load_pct"] / 100)
        dispatch["diesel_kw"]   = round(min(diesel, p["diesel_capacity_kw"]), 2)
        new_soc = bess_soc - (bess_discharge / p["bess_capacity_kwh"]) * 100

    # Pre-charge: if evening peak expected in next 6 h and SoC low
    if hour in range(10, 16) and bess_soc < 70 and solar_kw > 30:
        if dispatch["state"] == "PV_SURPLUS":
            dispatch["state"] = "PRE_CHARGE"

    dispatch["bess_soc_end"] = round(np.clip(new_soc, SEMS_PARAMS["bess_min_soc"], SEMS_PARAMS["bess_max_soc"]), 1)
    return dispatch


# ── Solar forecast (simple physics model) ────────────────────────────────────
def forecast_solar(dt_list: list[pd.Timestamp], capacity_kw: float, month: int) -> np.ndarray:
    """Estimate hourly solar generation from time of day and season."""
    is_harmattan = int(month in (11, 12, 1, 2))
    derating     = 1 - 0.25 * is_harmattan
    out = []
    for dt in dt_list:
        h = dt.hour
        if 6 <= h <= 18:
            peak_hour   = 13.0
            irr_fraction = np.exp(-((h - peak_hour) ** 2) / 10)
            gen          = capacity_kw * irr_fraction * derating
        else:
            gen = 0.0
        out.append(round(gen, 2))
    return np.array(out)


# ── Main prediction pipeline ──────────────────────────────────────────────────
def run_prediction(input_csv: str | None = None, current_soc: float = 60.0):
    now  = pd.Timestamp.now().floor("1h")
    date_str = now.strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"  MGIL Prediction — {now.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── Load models ──────────────────────────────────────────────────────────
    meta_path = os.path.join(MODELS_DIR, "meta.json")
    if not os.path.exists(meta_path):
        print("\n  ERROR: Models not found. Run mgil_train.py first.\n")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"\n[1/4] Loading models (trained {meta['trained_on'][:10]})...")
    lstm_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "lstm_model.keras"), compile=False
    )
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    # ── Get recent history (last 24 h) ───────────────────────────────────────
    print("\n[2/4] Loading recent load history...")
    if input_csv and os.path.exists(input_csv):
        raw = pd.read_csv(input_csv, index_col=0, parse_dates=True)
        raw.columns = raw.columns.str.lower()
        history_load = raw["load_kw"].tail(24).values
        print(f"      {len(history_load)} hours loaded from {input_csv}")
    else:
        # Try to load from the most recent forecast log
        log_files = sorted([
            f for f in os.listdir(FORECAST_DIR) if f.startswith("forecast_")
        ])
        if log_files:
            last_log = pd.read_csv(os.path.join(FORECAST_DIR, log_files[-1]))
            history_load = last_log["forecast_kw"].values[-24:]
            print(f"      Using last forecast as history seed: {log_files[-1]}")
        else:
            # Cold start: use synthetic profile for the current hour
            h = now.hour
            history_load = np.array([
                25 + 10 * np.sin(np.pi * (i - 6) / 12) * (6 <= i <= 18)
                + 15 * np.exp(-((i - 19) ** 2) / 4)
                + np.random.normal(0, 2)
                for i in range(h - 24, h)
            ]).clip(5)
            print("      Cold start: using synthetic history profile.")

    # Pad or trim to exactly 24
    if len(history_load) < 24:
        history_load = np.pad(history_load, (24 - len(history_load), 0), mode="edge")
    else:
        history_load = history_load[-24:]

    # ── 24-hour forecast ─────────────────────────────────────────────────────
    print("\n[3/4] Generating 24-hour load forecast...")
    forecast_times  = [now + pd.Timedelta(hours=i + 1) for i in range(24)]
    rf_forecasts    = []
    lstm_forecasts  = []
    rolling_history = list(history_load.copy())

    # LSTM: one shot from last 24h
    lstm_seq = scaler.transform(
        np.array(rolling_history[-24:]).reshape(-1, 1)
    ).flatten()

    for i, ft in enumerate(forecast_times):
        # RF forecast
        rf_feat = make_rf_features(np.array(rolling_history), ft)
        rf_pred = rf_model.predict(rf_feat)[0]

        # LSTM: slide window
        lstm_input = np.array(rolling_history[-24:])
        lstm_scaled = scaler.transform(lstm_input.reshape(-1, 1)).flatten()
        lstm_tensor = lstm_scaled.reshape(1, 24, 1)
        lstm_scaled_pred = lstm_model.predict(lstm_tensor, verbose=0)[0][0]
        lstm_pred = scaler.inverse_transform([[lstm_scaled_pred]])[0][0]

        # Ensemble: 40% LSTM, 60% RF
        ensemble = 0.4 * lstm_pred + 0.6 * rf_pred
        ensemble = max(5.0, ensemble)   # floor at 5 kW

        rf_forecasts.append(round(rf_pred, 2))
        lstm_forecasts.append(round(lstm_pred, 2))

        rolling_history.append(ensemble)

    ensemble_forecasts = [
        round(0.4 * lstm_forecasts[i] + 0.6 * rf_forecasts[i], 2)
        for i in range(24)
    ]

    # ── SEMS dispatch ────────────────────────────────────────────────────────
    print("\n[4/4] Running SEMS dispatch planning...")
    solar_forecasts = forecast_solar(
        forecast_times, SEMS_PARAMS["pv_capacity_kw"], now.month
    )

    dispatch_records = []
    bess_soc = current_soc

    for i, ft in enumerate(forecast_times):
        d = sems_dispatch(
            load_kw=ensemble_forecasts[i],
            solar_kw=float(solar_forecasts[i]),
            bess_soc=bess_soc,
            hour=ft.hour,
        )
        d["datetime"]       = ft.strftime("%Y-%m-%d %H:%M")
        d["rf_forecast_kw"] = rf_forecasts[i]
        d["lstm_forecast_kw"] = lstm_forecasts[i]
        d["forecast_kw"]    = ensemble_forecasts[i]
        bess_soc = d["bess_soc_end"]
        dispatch_records.append(d)

    # ── Save outputs ──────────────────────────────────────────────────────────
    dispatch_df = pd.DataFrame(dispatch_records)

    forecast_cols = ["datetime", "forecast_kw", "rf_forecast_kw", "lstm_forecast_kw",
                     "solar_kw", "bess_soc_start"]
    forecast_path = os.path.join(FORECAST_DIR, f"forecast_{date_str}.csv")
    dispatch_df[forecast_cols].to_csv(forecast_path, index=False)

    dispatch_path = os.path.join(FORECAST_DIR, f"dispatch_{date_str}.csv")
    dispatch_df.to_csv(dispatch_path, index=False)

    # ── Text report ───────────────────────────────────────────────────────────
    total_load   = dispatch_df["forecast_kw"].sum()
    total_solar  = dispatch_df["solar_kw"].sum()
    total_diesel = dispatch_df["diesel_kw"].sum()
    total_bess   = dispatch_df["bess_to_load"].sum()
    pv_fraction  = total_solar / total_load * 100 if total_load > 0 else 0
    peak_load    = dispatch_df["forecast_kw"].max()
    peak_hour    = dispatch_df.loc[dispatch_df["forecast_kw"].idxmax(), "datetime"]

    state_counts = dispatch_df["state"].value_counts()

    report = f"""MGIL Daily Forecast Report
===========================
Generated:  {now.strftime('%Y-%m-%d %H:%M')}
Horizon:    {forecast_times[0].strftime('%H:%M')} → {forecast_times[-1].strftime('%H:%M')}

24-Hour Energy Summary
----------------------
Projected load:        {total_load:.1f} kWh
Solar generation:      {total_solar:.1f} kWh  ({pv_fraction:.1f}% of load)
BESS contribution:     {total_bess:.1f} kWh
Diesel generation:     {total_diesel:.1f} kWh
Peak load:             {peak_load:.1f} kW  @ {peak_hour}

SEMS State Distribution (hours)
---------------------------------
"""
    for state, count in state_counts.items():
        report += f"  {state:<20} {count:>2} h\n"

    report += f"""
BESS Status
-----------
Starting SoC:   {current_soc:.1f}%
Ending SoC:     {dispatch_df['bess_soc_end'].iloc[-1]:.1f}%

Saved Files
-----------
{forecast_path}
{dispatch_path}
"""

    report_path = os.path.join(FORECAST_DIR, f"report_{date_str}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Print to console
    print(report)

    # ── Hourly dispatch table ─────────────────────────────────────────────────
    print("Hourly Dispatch Plan")
    print("-" * 72)
    print(f"{'Time':<8} {'Load kW':>8} {'Solar kW':>9} {'BESS kW':>8} "
          f"{'Diesel kW':>10} {'SoC%':>6}  State")
    print("-" * 72)
    for r in dispatch_records:
        bess_net = r["bess_to_load"] - r["pv_to_bess"]
        print(
            f"{r['datetime'][-5:]:<8} "
            f"{r['forecast_kw']:>8.1f} "
            f"{r['solar_kw']:>9.1f} "
            f"{bess_net:>8.1f} "
            f"{r['diesel_kw']:>10.1f} "
            f"{r['bess_soc_end']:>6.1f}  "
            f"{r['state']}"
        )
    print("-" * 72)
    print(f"\n  Forecast saved → {forecast_path}")
    print(f"  Dispatch saved → {dispatch_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MGIL daily load forecast and SEMS dispatch")
    parser.add_argument(
        "--input", metavar="CSV",
        help="CSV file with last 24h of load_kw readings (optional)"
    )
    parser.add_argument(
        "--soc", type=float, default=60.0, metavar="PERCENT",
        help="Current BESS state of charge in %% (default: 60.0)"
    )
    args = parser.parse_args()
    run_prediction(input_csv=args.input, current_soc=args.soc)
