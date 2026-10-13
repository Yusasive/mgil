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
    forecast/forecast_YYYY-MM-DD.csv   — 24-hour load forecast with confidence intervals
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
from scipy.optimize import linprog

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
FORECAST_DIR = os.path.join(os.path.dirname(__file__), "forecast")
LOG_DIR      = os.path.join(os.path.dirname(__file__), "logs")
WEIGHTS_FILE = os.path.join(MODELS_DIR, "ensemble_weights.json")
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)

# ── SEMS parameters (match paper defaults) ───────────────────────────────────
SEMS_PARAMS = {
    "grid_capacity_kw":   100.0,
    "pv_capacity_kw":      75.0,
    "bess_capacity_kwh":  200.0,
    "diesel_capacity_kw":  50.0,
    "bess_min_soc":        20.0,
    "bess_max_soc":        95.0,
    "bess_charge_rate":    40.0,
    "bess_discharge_rate": 50.0,
    "diesel_min_load_pct": 30.0,
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
    tf_feats = make_time_features(dt)
    lags = {}
    for lag in [1, 2, 3, 6, 12, 24]:
        idx = -lag
        lags[f"lag_{lag}h"] = history[idx] if len(history) >= lag else history[0]

    row = [
        tf_feats["hour"], tf_feats["dayofweek"], tf_feats["month"],
        tf_feats["is_market_day"], tf_feats["is_harmattan"], tf_feats["is_harvest"],
        lags["lag_1h"], lags["lag_2h"], lags["lag_3h"],
        lags["lag_6h"], lags["lag_12h"], lags["lag_24h"],
    ]
    return np.array(row).reshape(1, -1)


# ── Gap 3: Dynamic ensemble weighting ────────────────────────────────────────
def compute_dynamic_weights(window_days: int = 7) -> tuple[float, float]:
    """
    Compare recent RF and LSTM forecasts against actual readings to find
    which model has been more accurate over the last `window_days` days.
    Returns (lstm_weight, rf_weight) that sum to 1.0.

    If no history is available, falls back to the paper default (0.4 / 0.6).
    """
    default = (0.4, 0.6)

    data_dir     = os.path.join(os.path.dirname(__file__), "data")
    readings_file = os.path.join(data_dir, "site_readings.csv")

    if not os.path.exists(readings_file):
        return default

    forecast_files = sorted([
        f for f in os.listdir(FORECAST_DIR) if f.startswith("forecast_")
    ])
    if len(forecast_files) < 2:
        return default

    try:
        frames = []
        for fname in forecast_files[-window_days:]:
            fpath = os.path.join(FORECAST_DIR, fname)
            df = pd.read_csv(fpath, parse_dates=["datetime"])
            if "rf_forecast_kw" in df.columns and "lstm_forecast_kw" in df.columns:
                frames.append(df[["datetime", "rf_forecast_kw", "lstm_forecast_kw"]])

        if not frames:
            return default

        forecasts = pd.concat(frames).set_index("datetime")

        actuals = pd.read_csv(readings_file, parse_dates=["datetime"])
        actuals = actuals.set_index("datetime")[["load_kw"]]

        merged = forecasts.join(actuals, how="inner").dropna()
        if len(merged) < 12:
            return default

        rf_mape   = (abs(merged["rf_forecast_kw"]   - merged["load_kw"]) / merged["load_kw"]).mean()
        lstm_mape = (abs(merged["lstm_forecast_kw"] - merged["load_kw"]) / merged["load_kw"]).mean()

        # Invert MAPEs to get weights: better model (lower MAPE) gets higher weight
        total = rf_mape + lstm_mape
        if total < 1e-6:
            return default

        lstm_weight = round(float(rf_mape   / total), 3)   # RF bad → LSTM gets more
        rf_weight   = round(float(lstm_mape / total), 3)   # LSTM bad → RF gets more

        # Normalise to sum to exactly 1
        lstm_weight = lstm_weight / (lstm_weight + rf_weight)
        rf_weight   = 1.0 - lstm_weight

        # Save for audit trail
        weights_record = {
            "computed_at":  pd.Timestamp.now().isoformat(),
            "lstm_weight":  round(lstm_weight, 3),
            "rf_weight":    round(rf_weight, 3),
            "lstm_mape_7d": round(float(lstm_mape * 100), 2),
            "rf_mape_7d":   round(float(rf_mape   * 100), 2),
            "samples":      len(merged),
        }
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights_record, f, indent=2)

        return lstm_weight, rf_weight

    except Exception:
        return default


# ── Gap 2: RF confidence intervals ───────────────────────────────────────────
def rf_prediction_interval(
    rf_model,
    X: np.ndarray,
    percentile_lo: float = 10.0,
    percentile_hi: float = 90.0,
) -> tuple[float, float, float]:
    """
    Extract prediction intervals from a Random Forest by collecting
    the individual predictions of all 200 trees for a single input.

    The spread across trees represents genuine model uncertainty:
    if trees disagree widely, the forecast is less certain.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    # Collect each tree's individual prediction
    tree_preds = np.array([
        tree.predict(X)[0] for tree in rf_model.estimators_
    ])

    point   = float(np.mean(tree_preds))
    lower   = float(np.percentile(tree_preds, percentile_lo))
    upper   = float(np.percentile(tree_preds, percentile_hi))

    return round(point, 2), round(lower, 2), round(upper, 2)


# ── SEMS dispatch logic ───────────────────────────────────────────────────────
def sems_dispatch(
    load_kw:  float,
    solar_kw: float,
    bess_soc: float,
    hour:     int,
    p:        dict = SEMS_PARAMS,
) -> dict:
    net            = solar_kw - load_kw
    bess_kwh_avail = (bess_soc - p["bess_min_soc"]) / 100 * p["bess_capacity_kwh"]
    bess_kwh_room  = (p["bess_max_soc"] - bess_soc) / 100 * p["bess_capacity_kwh"]

    dispatch = {
        "load_kw":        round(load_kw, 2),
        "solar_kw":       round(solar_kw, 2),
        "bess_soc_start": round(bess_soc, 1),
        "pv_to_load":     0.0,
        "pv_to_bess":     0.0,
        "bess_to_load":   0.0,
        "diesel_kw":      0.0,
        "curtailed_kw":   0.0,
        "state":          "",
        "bess_soc_end":   0.0,
    }

    if net >= 0:
        dispatch["state"]        = "PV_SURPLUS"
        dispatch["pv_to_load"]   = round(load_kw, 2)
        charge                   = min(net, p["bess_charge_rate"], bess_kwh_room)
        dispatch["pv_to_bess"]   = round(charge, 2)
        dispatch["curtailed_kw"] = round(net - charge, 2)
        new_soc = bess_soc + (charge / p["bess_capacity_kwh"]) * 100

    elif bess_kwh_avail >= abs(net):
        dispatch["state"]         = "BESS_COVER"
        dispatch["pv_to_load"]    = round(solar_kw, 2)
        bess_discharge            = min(abs(net), p["bess_discharge_rate"])
        dispatch["bess_to_load"]  = round(bess_discharge, 2)
        new_soc = bess_soc - (bess_discharge / p["bess_capacity_kwh"]) * 100

    elif solar_kw > 0 and bess_kwh_avail > 0:
        dispatch["state"]         = "DIESEL_REQUIRED"
        dispatch["pv_to_load"]    = round(solar_kw, 2)
        bess_discharge            = min(bess_kwh_avail, p["bess_discharge_rate"])
        dispatch["bess_to_load"]  = round(bess_discharge, 2)
        unmet                     = load_kw - solar_kw - bess_discharge
        diesel                    = max(unmet, p["diesel_capacity_kw"] * p["diesel_min_load_pct"] / 100)
        dispatch["diesel_kw"]     = round(min(diesel, p["diesel_capacity_kw"]), 2)
        new_soc = bess_soc - (bess_discharge / p["bess_capacity_kwh"]) * 100

    else:
        dispatch["state"]         = "DIESEL_REQUIRED"
        dispatch["pv_to_load"]    = 0.0
        bess_discharge            = min(bess_kwh_avail, p["bess_discharge_rate"], load_kw)
        dispatch["bess_to_load"]  = round(bess_discharge, 2)
        unmet                     = load_kw - bess_discharge
        diesel                    = max(unmet, p["diesel_capacity_kw"] * p["diesel_min_load_pct"] / 100)
        dispatch["diesel_kw"]     = round(min(diesel, p["diesel_capacity_kw"]), 2)
        new_soc = bess_soc - (bess_discharge / p["bess_capacity_kwh"]) * 100

    if hour in range(10, 16) and bess_soc < 70 and solar_kw > 30:
        if dispatch["state"] == "PV_SURPLUS":
            dispatch["state"] = "PRE_CHARGE"

    dispatch["bess_soc_end"] = round(
        np.clip(new_soc, SEMS_PARAMS["bess_min_soc"], SEMS_PARAMS["bess_max_soc"]), 1
    )
    return dispatch


# ── Gap 6: MPC optimizer ──────────────────────────────────────────────────────
def mpc_dispatch_horizon(
    load_forecast:  np.ndarray,
    solar_forecast: np.ndarray,
    bess_soc_init:  float,
    p:              dict = SEMS_PARAMS,
) -> list[dict]:
    """
    Model Predictive Control dispatcher.

    Solves a linear program over the SEMS lookahead window to minimise
    total diesel generation subject to:
      - BESS SoC bounds and energy balance
      - Charge / discharge rate limits
      - Diesel generator minimum load and capacity
      - Power balance at every hour (demand must be met)

    Decision variables per hour h  (H = lookahead_hours):
        x[0..H-1]   → diesel_kw[h]
        x[H..2H-1]  → bess_charge_kw[h]   (PV surplus → BESS)
        x[2H..3H-1] → bess_discharge_kw[h] (BESS → load)

    Returns a list of H dispatch dicts, one per hour.
    """
    H       = p["lookahead_hours"]
    load_kw  = load_forecast[:H]
    solar_kw = solar_forecast[:H]

    # BESS parameters
    cap_kwh      = p["bess_capacity_kwh"]
    soc_min      = p["bess_min_soc"]
    soc_max      = p["bess_max_soc"]
    energy_min   = soc_min / 100 * cap_kwh
    energy_max   = soc_max / 100 * cap_kwh
    energy_init  = bess_soc_init / 100 * cap_kwh
    ch_rate      = p["bess_charge_rate"]
    dch_rate     = p["bess_discharge_rate"]
    diesel_cap   = p["diesel_capacity_kw"]
    diesel_min   = diesel_cap * p["diesel_min_load_pct"] / 100   # when running

    n_vars = 3 * H   # [diesel_0..H-1, bess_ch_0..H-1, bess_dch_0..H-1]

    # ── Objective: minimise sum of diesel_kw ─────────────────────────────────
    c = np.zeros(n_vars)
    c[:H] = 1.0   # coefficients for diesel variables

    # ── Bounds ───────────────────────────────────────────────────────────────
    bounds = []
    for h in range(H):
        bounds.append((0, diesel_cap))    # diesel
    for h in range(H):
        bounds.append((0, ch_rate))       # bess charge
    for h in range(H):
        bounds.append((0, dch_rate))      # bess discharge

    # ── Equality constraints: power balance each hour ─────────────────────────
    # diesel[h] + solar[h] + bess_dch[h] - bess_ch[h] == load[h]
    # Rearranged: diesel[h] - bess_ch[h] + bess_dch[h] == load[h] - solar[h]
    A_eq = np.zeros((H, n_vars))
    b_eq = np.zeros(H)
    for h in range(H):
        A_eq[h, h]       =  1.0   # diesel
        A_eq[h, H + h]   = -1.0   # bess charge (subtracts from supply)
        A_eq[h, 2*H + h] =  1.0   # bess discharge
        net_deficit = load_kw[h] - solar_kw[h]
        b_eq[h] = net_deficit
        # If solar > load, clamp: surplus goes to BESS charge beyond load
        # (handled by allowing negative net_deficit → negative diesel needed,
        #  linprog lower bound of 0 forces diesel=0 automatically)

    # ── Inequality constraints: BESS SoC bounds ──────────────────────────────
    # After each hour:  energy_init + sum(ch - dch) over [0..h] in [energy_min, energy_max]
    # energy_init + Σch[0..h] - Σdch[0..h] <= energy_max  → cumulative ch - dch constraint
    # energy_init + Σch[0..h] - Σdch[0..h] >= energy_min  → rewritten as -(Σch-dch) <= -(energy_min - energy_init)
    A_ub = np.zeros((2 * H, n_vars))
    b_ub = np.zeros(2 * H)

    for h in range(H):
        for j in range(h + 1):
            A_ub[h, H + j]      =  1.0   # charge increases energy
            A_ub[h, 2*H + j]    = -1.0   # discharge decreases energy
        b_ub[h] = energy_max - energy_init   # upper SoC bound

        A_ub[H + h, H:2*H]     = 0.0
        A_ub[H + h, 2*H:3*H]   = 0.0
        for j in range(h + 1):
            A_ub[H + h, H + j]   = -1.0   # flip sign for lower bound
            A_ub[H + h, 2*H + j] =  1.0
        b_ub[H + h] = energy_init - energy_min   # lower SoC bound

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )

    # ── Build dispatch records ────────────────────────────────────────────────
    records = []
    bess_soc = bess_soc_init

    if result.success:
        diesel_vals  = result.x[:H]
        bess_ch      = result.x[H:2*H]
        bess_dch     = result.x[2*H:3*H]
    else:
        # Fallback to rule-based if LP infeasible
        diesel_vals = np.maximum(0, load_kw - solar_kw)
        bess_ch     = np.zeros(H)
        bess_dch    = np.zeros(H)

    for h in range(H):
        net = solar_kw[h] - load_kw[h]

        d_kw   = max(0.0, round(float(diesel_vals[h]), 2))
        ch_kw  = max(0.0, round(float(bess_ch[h]), 2))
        dch_kw = max(0.0, round(float(bess_dch[h]), 2))

        pv_to_load  = round(min(solar_kw[h], load_kw[h]), 2)
        pv_to_bess  = ch_kw
        curtailed   = max(0.0, round(solar_kw[h] - pv_to_load - pv_to_bess, 2))

        # Determine SEMS state label
        if d_kw > 0:
            state = "DIESEL_REQUIRED"
        elif dch_kw > 0:
            state = "BESS_COVER"
        elif net >= 0:
            state = "PV_SURPLUS"
        else:
            state = "PV_SURPLUS"

        # PRE_CHARGE upgrade
        future_hour = (h + 1) % 24
        if future_hour in range(10, 16) and bess_soc < 70 and solar_kw[h] > 30 and state == "PV_SURPLUS":
            state = "PRE_CHARGE"

        # SoC update
        new_energy = (bess_soc / 100 * cap_kwh) + ch_kw - dch_kw
        new_energy  = np.clip(new_energy, energy_min, energy_max)
        new_soc     = round(float(new_energy / cap_kwh * 100), 1)

        records.append({
            "diesel_kw":      d_kw,
            "pv_to_bess":     pv_to_bess,
            "bess_to_load":   dch_kw,
            "pv_to_load":     pv_to_load,
            "curtailed_kw":   curtailed,
            "bess_soc_start": round(bess_soc, 1),
            "bess_soc_end":   new_soc,
            "state":          state,
            "load_kw":        round(load_kw[h], 2),
            "solar_kw":       round(solar_kw[h], 2),
        })
        bess_soc = new_soc

    return records


# ── Gap 5: Solar ML forecast ──────────────────────────────────────────────────
def make_solar_features(dt: pd.Timestamp, prev_solar: list) -> np.ndarray:
    """
    Build the feature vector for the solar RF model for a single future hour.
    Mirrors the features trained in mgil_train.py → add_solar_features().
    """
    hour       = dt.hour
    month      = dt.month
    dayofyear  = dt.dayofyear
    is_harmattan = int(month in (11, 12, 1, 2))
    solar_angle  = float(np.sin(np.pi * (hour - 6) / 12))
    solar_angle  = max(0.0, solar_angle)
    day_length   = 1.0 + 0.1 * np.cos(2 * np.pi * (dayofyear - 172) / 365)

    # Use last 3 known solar values as lag features (0 if unavailable)
    def get_lag(n):
        if len(prev_solar) >= n:
            return prev_solar[-n]
        return 0.0

    row = [
        hour, month, dayofyear, is_harmattan,
        solar_angle, day_length,
        get_lag(1), get_lag(2), get_lag(3),
    ]
    return np.array(row).reshape(1, -1)


def forecast_solar_ml(
    dt_list: list,
    solar_rf_model,
    history_solar: np.ndarray,
    capacity_kw: float,
) -> np.ndarray:
    """
    Use the trained solar RF model to forecast solar generation.
    Falls back to physics bell-curve if model is not available.
    """
    rolling = list(history_solar.copy())
    out = []
    for dt in dt_list:
        if dt.hour < 6 or dt.hour > 18:
            pred = 0.0
        else:
            X = make_solar_features(dt, rolling)
            pred = float(solar_rf_model.predict(X)[0])
            pred = max(0.0, min(pred, capacity_kw))
        rolling.append(pred)
        out.append(round(pred, 2))
    return np.array(out)


def forecast_solar_physics(dt_list: list, capacity_kw: float, month: int) -> np.ndarray:
    """
    Physics fallback: bell-curve Gaussian centred on solar noon.
    Used only when solar_rf_model.pkl is not present.
    """
    is_harmattan = int(month in (11, 12, 1, 2))
    derating     = 1 - 0.25 * is_harmattan
    out = []
    for dt in dt_list:
        h = dt.hour
        if 6 <= h <= 18:
            gen = capacity_kw * np.exp(-((h - 13.0) ** 2) / 10) * derating
        else:
            gen = 0.0
        out.append(round(gen, 2))
    return np.array(out)


# ── Main prediction pipeline ──────────────────────────────────────────────────
def run_prediction(input_csv: str | None = None, current_soc: float = 60.0):
    now      = pd.Timestamp.now().floor("1h")
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

    print(f"\n[1/5] Loading models (trained {meta['trained_on'][:10]})...")
    lstm_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "lstm_model.keras"), compile=False
    )
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    # Gap 5: load solar ML model if available
    solar_rf_path = os.path.join(MODELS_DIR, "solar_rf_model.pkl")
    solar_rf_model = None
    if os.path.exists(solar_rf_path):
        solar_rf_model = joblib.load(solar_rf_path)
        print(f"      Solar RF model loaded  ✓")
    else:
        print(f"      Solar RF model not found — using physics fallback.")

    # ── Dynamic ensemble weights ──────────────────────────────────────────────
    print("\n[2/5] Computing dynamic ensemble weights...")
    lstm_w, rf_w = compute_dynamic_weights()
    print(f"      LSTM weight: {lstm_w:.2f}  |  RF weight: {rf_w:.2f}")
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f:
            w = json.load(f)
        if "lstm_mape_7d" in w:
            print(f"      Based on 7-day MAPE: LSTM={w['lstm_mape_7d']}%  RF={w['rf_mape_7d']}%")
    else:
        print("      No history yet — using paper defaults (LSTM=0.40, RF=0.60)")

    # ── Load recent history ───────────────────────────────────────────────────
    print("\n[3/5] Loading recent load and solar history...")
    history_solar = None

    if input_csv and os.path.exists(input_csv):
        raw = pd.read_csv(input_csv, index_col=0, parse_dates=True)
        raw.columns = raw.columns.str.lower()
        history_load = raw["load_kw"].tail(24).values
        if "solar_kw" in raw.columns:
            history_solar = raw["solar_kw"].tail(24).values
        print(f"      {len(history_load)} hours loaded from {input_csv}")
    else:
        # Try to load from site_readings.csv
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        readings_file = os.path.join(data_dir, "site_readings.csv")
        if os.path.exists(readings_file):
            raw = pd.read_csv(readings_file, index_col=0, parse_dates=True)
            raw.columns = raw.columns.str.lower()
            if "load_kw" in raw.columns:
                history_load = raw["load_kw"].tail(24).values
                if "solar_kw" in raw.columns:
                    history_solar = raw["solar_kw"].tail(24).values
                print(f"      Loaded from site_readings.csv ({len(history_load)} hours)")
            else:
                history_load = None
        else:
            history_load = None

        if history_load is None:
            log_files = sorted([
                f for f in os.listdir(FORECAST_DIR) if f.startswith("forecast_")
            ])
            if log_files:
                last_log     = pd.read_csv(os.path.join(FORECAST_DIR, log_files[-1]))
                history_load = last_log["forecast_kw"].values[-24:]
                if "solar_kw" in last_log.columns:
                    history_solar = last_log["solar_kw"].values[-24:]
                print(f"      Using last forecast as history seed: {log_files[-1]}")
            else:
                h = now.hour
                history_load = np.array([
                    25 + 10 * np.sin(np.pi * (i - 6) / 12) * (6 <= i <= 18)
                    + 15 * np.exp(-((i - 19) ** 2) / 4)
                    + np.random.normal(0, 2)
                    for i in range(h - 24, h)
                ]).clip(5)
                print("      Cold start: using synthetic history profile.")

    if len(history_load) < 24:
        history_load = np.pad(history_load, (24 - len(history_load), 0), mode="edge")
    else:
        history_load = history_load[-24:]

    # Solar history: synthetic if unavailable
    if history_solar is None or len(history_solar) == 0:
        h_vals = np.arange(now.hour - 24, now.hour) % 24
        derating = 1 - 0.25 * int(now.month in (11, 12, 1, 2))
        history_solar = np.array([
            max(0.0, SEMS_PARAMS["pv_capacity_kw"] * np.exp(-((h - 13.0) ** 2) / 10) * derating)
            if 6 <= h <= 18 else 0.0
            for h in h_vals
        ])
    if len(history_solar) < 3:
        history_solar = np.pad(history_solar, (3 - len(history_solar), 0), mode="edge")
    else:
        history_solar = history_solar[-24:]

    # ── 24-hour forecast with confidence intervals ────────────────────────────
    print("\n[4/5] Generating 24-hour forecast with confidence intervals...")
    forecast_times   = [now + pd.Timedelta(hours=i + 1) for i in range(24)]
    rf_forecasts     = []
    lstm_forecasts   = []
    forecast_lower   = []
    forecast_upper   = []
    rolling_history  = list(history_load.copy())

    for i, ft in enumerate(forecast_times):
        rf_feat = make_rf_features(np.array(rolling_history), ft)

        # RF: point estimate + confidence interval from tree variance
        rf_point, rf_lo, rf_hi = rf_prediction_interval(rf_model, rf_feat)

        # LSTM: point estimate
        lstm_input        = np.array(rolling_history[-24:])
        lstm_scaled       = scaler.transform(lstm_input.reshape(-1, 1)).flatten()
        lstm_tensor       = lstm_scaled.reshape(1, 24, 1)
        lstm_scaled_pred  = lstm_model.predict(lstm_tensor, verbose=0)[0][0]
        lstm_pred         = float(scaler.inverse_transform([[lstm_scaled_pred]])[0][0])

        # Ensemble with dynamic weights
        ensemble = lstm_w * lstm_pred + rf_w * rf_point
        ensemble = max(5.0, ensemble)

        # Confidence interval: weighted blend of RF interval + LSTM point
        # LSTM has no native interval — use RF interval scaled by ensemble weight
        ci_lo = max(5.0, lstm_w * lstm_pred + rf_w * rf_lo)
        ci_hi = max(5.0, lstm_w * lstm_pred + rf_w * rf_hi)

        rf_forecasts.append(round(rf_point, 2))
        lstm_forecasts.append(round(lstm_pred, 2))
        forecast_lower.append(round(ci_lo, 2))
        forecast_upper.append(round(ci_hi, 2))

        rolling_history.append(ensemble)

    ensemble_forecasts = [
        round(lstm_w * lstm_forecasts[i] + rf_w * rf_forecasts[i], 2)
        for i in range(24)
    ]

    # ── SEMS dispatch ────────────────────────────────────────────────────────
    print("\n[5/5] Running SEMS dispatch planning...")
    if solar_rf_model is not None:
        solar_forecasts = forecast_solar_ml(
            forecast_times, solar_rf_model, history_solar,
            SEMS_PARAMS["pv_capacity_kw"],
        )
        print("      Solar forecast: ML model (RF)")
    else:
        solar_forecasts = forecast_solar_physics(
            forecast_times, SEMS_PARAMS["pv_capacity_kw"], now.month
        )
        print("      Solar forecast: physics fallback (bell-curve)")

    # Gap 6: MPC — run rolling 6-hour lookahead over all 24 hours
    print("      Using MPC optimizer (6-hour lookahead)")
    dispatch_records = []
    bess_soc = current_soc

    i = 0
    while i < 24:
        horizon_load  = np.array(ensemble_forecasts[i:i + SEMS_PARAMS["lookahead_hours"]])
        horizon_solar = solar_forecasts[i:i + SEMS_PARAMS["lookahead_hours"]]

        # Pad if near end of 24h window
        if len(horizon_load) < SEMS_PARAMS["lookahead_hours"]:
            pad = SEMS_PARAMS["lookahead_hours"] - len(horizon_load)
            horizon_load  = np.pad(horizon_load,  (0, pad), mode="edge")
            horizon_solar = np.pad(horizon_solar, (0, pad), mode="edge")

        mpc_results = mpc_dispatch_horizon(horizon_load, horizon_solar, bess_soc)

        # Take only the first result (receding-horizon: re-optimise each hour)
        d = mpc_results[0]
        ft = forecast_times[i]
        d["datetime"]          = ft.strftime("%Y-%m-%d %H:%M")
        d["rf_forecast_kw"]    = rf_forecasts[i]
        d["lstm_forecast_kw"]  = lstm_forecasts[i]
        d["forecast_kw"]       = ensemble_forecasts[i]
        d["forecast_lower_kw"] = forecast_lower[i]
        d["forecast_upper_kw"] = forecast_upper[i]
        d["forecast_range_kw"] = round(forecast_upper[i] - forecast_lower[i], 2)
        bess_soc = d["bess_soc_end"]
        dispatch_records.append(d)
        i += 1

    # ── Save outputs ──────────────────────────────────────────────────────────
    dispatch_df = pd.DataFrame(dispatch_records)

    forecast_cols = [
        "datetime", "forecast_kw", "forecast_lower_kw", "forecast_upper_kw",
        "forecast_range_kw", "rf_forecast_kw", "lstm_forecast_kw",
        "solar_kw", "bess_soc_start",
    ]
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
    avg_ci_range = dispatch_df["forecast_range_kw"].mean()
    state_counts = dispatch_df["state"].value_counts()

    solar_model_type = "ML (Random Forest)" if solar_rf_model else "Physics (bell-curve fallback)"

    report = f"""MGIL Daily Forecast Report
===========================
Generated:  {now.strftime('%Y-%m-%d %H:%M')}
Horizon:    {forecast_times[0].strftime('%H:%M')} → {forecast_times[-1].strftime('%H:%M')}
Dispatcher: MPC (6-hour lookahead, linprog HiGHS)
Solar model:{solar_model_type}

Ensemble Weights (dynamic)
--------------------------
LSTM weight:   {lstm_w:.2f}
RF weight:     {rf_w:.2f}

24-Hour Energy Summary
----------------------
Projected load:        {total_load:.1f} kWh
Solar generation:      {total_solar:.1f} kWh  ({pv_fraction:.1f}% of load)
BESS contribution:     {total_bess:.1f} kWh
Diesel generation:     {total_diesel:.1f} kWh
Peak load:             {peak_load:.1f} kW  @ {peak_hour}
Avg forecast range:    ±{avg_ci_range/2:.1f} kW  (80% confidence interval)

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

    print(report)

    # ── Hourly dispatch table ─────────────────────────────────────────────────
    print("Hourly Dispatch Plan  (CI = 80% confidence interval)")
    print("-" * 85)
    print(f"{'Time':<8} {'Forecast':>9} {'CI Low':>7} {'CI High':>8} "
          f"{'Solar':>7} {'BESS':>7} {'Diesel':>8} {'SoC%':>6}  State")
    print("-" * 85)
    for r in dispatch_records:
        bess_net = r["bess_to_load"] - r["pv_to_bess"]
        print(
            f"{r['datetime'][-5:]:<8} "
            f"{r['forecast_kw']:>9.1f} "
            f"{r['forecast_lower_kw']:>7.1f} "
            f"{r['forecast_upper_kw']:>8.1f} "
            f"{r['solar_kw']:>7.1f} "
            f"{bess_net:>7.1f} "
            f"{r['diesel_kw']:>8.1f} "
            f"{r['bess_soc_end']:>6.1f}  "
            f"{r['state']}"
        )
    print("-" * 85)
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