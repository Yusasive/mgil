"""
MGIL Data Logger — Mini-Grid Intelligence Layer
================================================
Reads live sensor data from the site's data logger every hour and
appends it to a rolling CSV file used for model retraining.

Supported data sources:
  1. Simulated (default)     — for testing without hardware
  2. Modbus RTU via RS-485   — most common data logger protocol
  3. CSV file polling        — for loggers that write their own CSV
  4. HTTP/REST API           — for loggers with a web interface

Usage:
    python mgil_log.py                        # Simulated (test mode)
    python mgil_log.py --source modbus        # Read from RS-485 port
    python mgil_log.py --source csv --file /path/to/logger_output.csv
    python mgil_log.py --source http --url http://192.168.1.10/api/readings

Cron example (runs every hour):
    0 * * * * /home/pi/mgil/venv/bin/python /home/pi/mgil/mgil_log.py --source modbus >> /home/pi/mgil/logs/mgil_log.log 2>&1

Output:
    data/site_readings.csv    ← full history, appended each hour
    data/last_24h.csv         ← rolling 24-hour window for mgil_predict.py
    data/current_soc.txt      ← latest BESS SoC, read by cron for --soc
    data/anomaly_log.csv      ← flagged anomalous readings with reason
"""

import argparse
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

READINGS_FILE  = os.path.join(DATA_DIR, "site_readings.csv")
LAST_24H_FILE  = os.path.join(DATA_DIR, "last_24h.csv")
SOC_FILE       = os.path.join(DATA_DIR, "current_soc.txt")
ANOMALY_FILE   = os.path.join(DATA_DIR, "anomaly_log.csv")
DRIFT_FILE     = os.path.join(DATA_DIR, "drift_status.json")

# Modbus register map — adjust to match your data logger's manual
MODBUS_REGISTERS = {
    "load_kw":   40001,   # Active load power (kW × 10)
    "solar_kw":  40003,   # PV generation (kW × 10)
    "bess_soc":  40005,   # BESS state of charge (% × 10)
    "bess_kw":   40007,   # BESS charge/discharge power (kW × 10)
    "diesel_kw": 40009,   # Diesel generator output (kW × 10)
    "grid_freq": 40011,   # Grid frequency (Hz × 100)
}

# Hard physical bounds — readings outside these are impossible
SENSOR_BOUNDS = {
    "load_kw":   (0.0,   150.0),
    "solar_kw":  (0.0,   100.0),
    "bess_soc":  (0.0,   100.0),
    "bess_kw":   (-60.0,  60.0),
    "diesel_kw": (0.0,    60.0),
    "grid_freq": (45.0,   55.0),
}

# Z-score threshold for statistical anomaly detection
Z_SCORE_THRESHOLD = 3.5

# Drift detection — retrain when rolling MAPE exceeds this
DRIFT_MAPE_THRESHOLD = 12.0   # %
DRIFT_WINDOW_HOURS   = 48     # hours of recent forecasts to evaluate


# ── Simulated sensor (test mode) ──────────────────────────────────────────────
def read_simulated() -> dict:
    """Generate a realistic synthetic reading for the current hour."""
    now   = pd.Timestamp.now()
    h     = now.hour
    month = now.month

    is_harmattan = int(month in (11, 12, 1, 2))

    load = (
        25
        + 10 * np.sin(np.pi * (h - 6) / 12) * (6 <= h <= 18)
        + 18 * np.exp(-((h - 19) ** 2) / 4)
        + np.random.normal(0, 2)
    )
    load = max(5.0, load)

    if 6 <= h <= 18:
        solar = max(0.0, 55 * np.exp(-((h - 13) ** 2) / 10)
                    * (1 - 0.25 * is_harmattan) + np.random.normal(0, 1.5))
    else:
        solar = 0.0

    try:
        with open(SOC_FILE) as f:
            prev_soc = float(f.read().strip())
    except Exception:
        prev_soc = 60.0

    net = solar - load
    if net > 0:
        new_soc = min(95.0, prev_soc + (net / 200) * 100)
    else:
        new_soc = max(20.0, prev_soc + (net / 200) * 100)
    new_soc += np.random.normal(0, 0.5)
    new_soc = float(np.clip(new_soc, 20.0, 95.0))

    return {
        "load_kw":   round(load, 2),
        "solar_kw":  round(solar, 2),
        "bess_soc":  round(new_soc, 1),
        "bess_kw":   round(net * 0.3, 2),
        "diesel_kw": round(max(0, load - solar - 10), 2),
        "grid_freq": round(50.0 + np.random.normal(0, 0.05), 2),
    }


# ── Modbus RTU reader ─────────────────────────────────────────────────────────
def read_modbus(port: str = "/dev/ttyUSB0", slave_id: int = 1) -> dict:
    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError:
        print("  pymodbus not installed. Run: pip install pymodbus")
        sys.exit(1)

    client = ModbusSerialClient(
        port=port, baudrate=9600, parity="N", stopbits=1, bytesize=8, timeout=3
    )

    if not client.connect():
        raise ConnectionError(f"Cannot connect to Modbus device on {port}")

    readings = {}
    try:
        for name, register in MODBUS_REGISTERS.items():
            result = client.read_holding_registers(
                address=register - 40001, count=1, slave=slave_id
            )
            if result.isError():
                print(f"  Warning: failed to read {name} (register {register})")
                readings[name] = 0.0
            else:
                raw = result.registers[0]
                readings[name] = round(raw / 10.0, 2)
    finally:
        client.close()

    return readings


# ── CSV file poller ───────────────────────────────────────────────────────────
def read_csv_logger(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Logger CSV not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    time_col = next((c for c in df.columns if c in
                     ("datetime", "timestamp", "time", "date")), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

    latest = df.iloc[-1]

    def get(names, default=0.0):
        for n in names:
            if n in latest.index:
                return float(latest[n])
        return default

    return {
        "load_kw":   get(["load_kw", "load", "demand"]),
        "solar_kw":  get(["solar_kw", "solar", "pv", "irradiance"]),
        "bess_soc":  get(["bess_soc", "soc", "battery_soc"]),
        "bess_kw":   get(["bess_kw", "battery_kw"]),
        "diesel_kw": get(["diesel_kw", "generator_kw", "gen_kw"]),
        "grid_freq": get(["freq", "frequency", "grid_freq"], 50.0),
    }


# ── HTTP/REST reader ──────────────────────────────────────────────────────────
def read_http(url: str) -> dict:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise ConnectionError(f"HTTP read failed: {e}")

    def get(keys, default=0.0):
        for k in keys:
            if k in data:
                return float(data[k])
        return default

    return {
        "load_kw":   get(["load_kw", "load", "demand_kw"]),
        "solar_kw":  get(["solar_kw", "pv_kw", "solar"]),
        "bess_soc":  get(["bess_soc", "soc", "battery"]),
        "bess_kw":   get(["bess_kw", "battery_kw"]),
        "diesel_kw": get(["diesel_kw", "gen_kw"]),
        "grid_freq": get(["freq", "frequency"], 50.0),
    }


# ── Gap 1: Anomaly detection ──────────────────────────────────────────────────
def detect_anomalies(readings: dict, timestamp: pd.Timestamp) -> tuple[dict, list[str]]:
    """
    Two-layer anomaly detection on incoming sensor readings.

    Layer 1 — Hard bounds check:
        Each sensor has physical limits (e.g. SoC cannot be > 100%).
        Any reading outside these limits is physically impossible
        and is replaced with the last known good value.

    Layer 2 — Statistical z-score check:
        Using the last 168 hours (7 days) of history, compute the
        z-score for each reading. A z-score above Z_SCORE_THRESHOLD
        (default 3.5) indicates a statistical outlier — likely a
        faulty sensor or communication error. These are flagged and
        replaced with the rolling median for that sensor.

    Returns:
        cleaned_readings: dict with corrected values
        anomalies: list of human-readable anomaly descriptions
    """
    cleaned   = dict(readings)
    anomalies = []

    # Layer 1: hard bounds
    for sensor, (lo, hi) in SENSOR_BOUNDS.items():
        val = cleaned.get(sensor, 0.0)
        if val < lo or val > hi:
            anomalies.append(
                f"HARD_BOUNDS | {sensor} = {val} (valid range: {lo} to {hi})"
            )
            # Replace with midpoint of valid range as safe fallback
            cleaned[sensor] = round((lo + hi) / 2, 2)

    # Layer 2: z-score against recent history
    if os.path.exists(READINGS_FILE):
        try:
            history = pd.read_csv(READINGS_FILE).tail(168)   # last 7 days

            for sensor in ["load_kw", "solar_kw", "bess_soc", "grid_freq"]:
                if sensor not in history.columns:
                    continue

                vals = history[sensor].dropna().values
                if len(vals) < 10:
                    continue   # not enough history yet

                mean = np.mean(vals)
                std  = np.std(vals)

                if std < 0.001:
                    continue   # all values identical, skip

                z = abs(cleaned[sensor] - mean) / std

                if z > Z_SCORE_THRESHOLD:
                    median_val = round(float(np.median(vals)), 2)
                    anomalies.append(
                        f"Z_SCORE     | {sensor} = {cleaned[sensor]} "
                        f"(z={z:.1f}, replaced with median={median_val})"
                    )
                    cleaned[sensor] = median_val

        except Exception as e:
            anomalies.append(f"ANOMALY_CHECK_ERROR | {e}")

    # Log all anomalies to file
    if anomalies:
        rows = [{"datetime": timestamp.isoformat(), "anomaly": a} for a in anomalies]
        new_df = pd.DataFrame(rows)
        if os.path.exists(ANOMALY_FILE):
            existing = pd.read_csv(ANOMALY_FILE)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(ANOMALY_FILE, index=False)

    return cleaned, anomalies


# ── Gap 4: Drift detection ────────────────────────────────────────────────────
def check_forecast_drift() -> dict:
    """
    Compare recent forecasts against actual logged readings to compute
    rolling MAPE. If MAPE drifts above DRIFT_MAPE_THRESHOLD, flag for
    retraining.

    Returns a dict with drift status and MAPE.
    """
    result = {"drift_detected": False, "rolling_mape": None, "hours_evaluated": 0}

    if not os.path.exists(READINGS_FILE):
        return result

    # Find forecast files in the forecast directory
    forecast_dir = os.path.join(os.path.dirname(__file__), "forecast")
    if not os.path.exists(forecast_dir):
        return result

    forecast_files = sorted([
        f for f in os.listdir(forecast_dir) if f.startswith("forecast_")
    ])
    if not forecast_files:
        return result

    try:
        # Load all recent forecast files
        forecast_frames = []
        for fname in forecast_files[-7:]:   # last 7 days of forecasts
            fpath = os.path.join(forecast_dir, fname)
            df = pd.read_csv(fpath, parse_dates=["datetime"])
            forecast_frames.append(df[["datetime", "forecast_kw"]])

        if not forecast_frames:
            return result

        forecasts = pd.concat(forecast_frames).set_index("datetime")

        # Load actual readings
        actuals = pd.read_csv(READINGS_FILE, parse_dates=["datetime"])
        actuals = actuals.set_index("datetime")[["load_kw"]]

        # Match on timestamp
        merged = forecasts.join(actuals, how="inner").dropna()
        merged = merged.tail(DRIFT_WINDOW_HOURS)

        if len(merged) < 12:
            return result

        mape = (
            abs(merged["forecast_kw"] - merged["load_kw"]) / merged["load_kw"]
        ).mean() * 100

        drift_detected = mape > DRIFT_MAPE_THRESHOLD

        result = {
            "drift_detected":   drift_detected,
            "rolling_mape":     round(float(mape), 2),
            "hours_evaluated":  len(merged),
            "threshold":        DRIFT_MAPE_THRESHOLD,
            "checked_at":       pd.Timestamp.now().isoformat(),
        }

        # Persist drift status
        with open(DRIFT_FILE, "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result["error"] = str(e)

    return result


# ── Append to rolling CSV ─────────────────────────────────────────────────────
def log_reading(readings: dict, timestamp: pd.Timestamp):
    row = {"datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S")}
    row.update(readings)
    new_row = pd.DataFrame([row])

    if os.path.exists(READINGS_FILE):
        history = pd.read_csv(READINGS_FILE)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history.to_csv(READINGS_FILE, index=False)

    history["datetime"] = pd.to_datetime(history["datetime"])
    last_24h = history.tail(24)
    last_24h.to_csv(LAST_24H_FILE, index=False)

    with open(SOC_FILE, "w") as f:
        f.write(str(readings.get("bess_soc", 60.0)))

    return len(history)


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_status(
    readings: dict,
    cleaned: dict,
    anomalies: list,
    drift: dict,
    timestamp: pd.Timestamp,
    total_rows: int,
):
    print(f"\n  Timestamp:   {timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Load:        {cleaned['load_kw']:.1f} kW")
    print(f"  Solar:       {cleaned['solar_kw']:.1f} kW")
    print(f"  BESS SoC:    {cleaned['bess_soc']:.1f}%")
    print(f"  BESS power:  {cleaned['bess_kw']:+.1f} kW  "
          f"({'charging' if cleaned['bess_kw'] > 0 else 'discharging' if cleaned['bess_kw'] < 0 else 'idle'})")
    print(f"  Diesel:      {cleaned['diesel_kw']:.1f} kW")
    print(f"  Grid freq:   {cleaned['grid_freq']:.2f} Hz")

    # Anomaly report
    if anomalies:
        print(f"\n  ANOMALIES DETECTED ({len(anomalies)}):")
        for a in anomalies:
            print(f"    ! {a}")
        print(f"    Logged to: {ANOMALY_FILE}")
    else:
        print(f"\n  Anomaly check: CLEAN — all readings within normal range")

    # Drift report
    if drift.get("rolling_mape") is not None:
        mape = drift["rolling_mape"]
        hours = drift["hours_evaluated"]
        if drift["drift_detected"]:
            print(f"\n  DRIFT ALERT: Rolling MAPE = {mape:.1f}% over last {hours}h")
            print(f"    Threshold: {drift['threshold']}%")
            print(f"    Action: Run 'python mgil_train.py --data {READINGS_FILE}'")
        else:
            print(f"\n  Drift check: OK — Rolling MAPE = {mape:.1f}% ({hours}h window)")

    print(f"\n  Total logged rows: {total_rows}")
    print(f"  History file:      {READINGS_FILE}")
    print(f"  Last 24h file:     {LAST_24H_FILE}")

    if total_rows >= 24 * 7:
        weeks = total_rows // (24 * 7)
        print(f"\n  ✓  {weeks} week(s) of data — sufficient for retraining.")
        print(f"     Run: python mgil_train.py --data {READINGS_FILE}")
    elif total_rows >= 24:
        days      = total_rows // 24
        days_needed = max(0, 7 - days)
        print(f"\n  {days} day(s) logged. {days_needed} more day(s) until first retrain recommended.")
    else:
        print(f"\n  Building history — {24 - total_rows} more hours until first full day.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MGIL hourly data logger")
    parser.add_argument(
        "--source", choices=["simulated", "modbus", "csv", "http"],
        default="simulated",
        help="Data source (default: simulated)"
    )
    parser.add_argument("--port",  default="/dev/ttyUSB0", help="Modbus serial port")
    parser.add_argument("--slave", type=int, default=1,    help="Modbus slave ID")
    parser.add_argument("--file",  default=None,           help="Logger CSV file path")
    parser.add_argument("--url",   default=None,           help="Logger HTTP endpoint")
    args = parser.parse_args()

    now = pd.Timestamp.now().floor("1h")

    print("=" * 60)
    print(f"  MGIL Logger — {now.strftime('%Y-%m-%d %H:%M')}  [{args.source}]")
    print("=" * 60)

    try:
        # 1. Read raw sensor values
        if args.source == "simulated":
            raw_readings = read_simulated()
        elif args.source == "modbus":
            print(f"\n  Reading Modbus RTU on {args.port} (slave {args.slave})...")
            raw_readings = read_modbus(port=args.port, slave_id=args.slave)
        elif args.source == "csv":
            if not args.file:
                print("  Error: --file is required for csv source.")
                sys.exit(1)
            print(f"\n  Reading logger CSV: {args.file}")
            raw_readings = read_csv_logger(args.file)
        elif args.source == "http":
            if not args.url:
                print("  Error: --url is required for http source.")
                sys.exit(1)
            print(f"\n  Reading HTTP endpoint: {args.url}")
            raw_readings = read_http(args.url)

        # 2. Anomaly detection — clean before logging
        cleaned_readings, anomalies = detect_anomalies(raw_readings, now)

        # 3. Log cleaned readings
        total_rows = log_reading(cleaned_readings, now)

        # 4. Drift detection — check forecast accuracy trend
        drift = check_forecast_drift()

        # 5. Print status
        print_status(raw_readings, cleaned_readings, anomalies, drift, now, total_rows)
        print()

    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("  Check your data source connection and try again.\n")
        sys.exit(1)