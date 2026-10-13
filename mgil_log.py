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
"""

import argparse
import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

READINGS_FILE = os.path.join(DATA_DIR, "site_readings.csv")
LAST_24H_FILE = os.path.join(DATA_DIR, "last_24h.csv")
SOC_FILE      = os.path.join(DATA_DIR, "current_soc.txt")

# Modbus register map — adjust to match your data logger's manual
MODBUS_REGISTERS = {
    "load_kw":   40001,   # Active load power (kW × 10)
    "solar_kw":  40003,   # PV generation (kW × 10)
    "bess_soc":  40005,   # BESS state of charge (% × 10)
    "bess_kw":   40007,   # BESS charge/discharge power (kW × 10)
    "diesel_kw":40009,    # Diesel generator output (kW × 10)
    "grid_freq": 40011,   # Grid frequency (Hz × 100)
}


# ── Simulated sensor (test mode) ──────────────────────────────────────────────
def read_simulated() -> dict:
    """Generate a realistic synthetic reading for the current hour."""
    now   = pd.Timestamp.now()
    h     = now.hour
    month = now.month

    is_harmattan = int(month in (11, 12, 1, 2))

    # Load
    load = (
        25
        + 10 * np.sin(np.pi * (h - 6) / 12) * (6 <= h <= 18)
        + 18 * np.exp(-((h - 19) ** 2) / 4)
        + np.random.normal(0, 2)
    )
    load = max(5.0, load)

    # Solar
    if 6 <= h <= 18:
        solar = max(0.0, 55 * np.exp(-((h - 13) ** 2) / 10)
                    * (1 - 0.25 * is_harmattan) + np.random.normal(0, 1.5))
    else:
        solar = 0.0

    # SoC: read from last known value if available, else default
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
    """
    Read registers from a Modbus RTU data logger over RS-485.
    Requires: pip install pymodbus
    Common port names: /dev/ttyUSB0 (Raspberry Pi), COM3 (Windows)
    """
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
                # Most loggers scale by ×10; adjust if yours differs
                readings[name] = round(raw / 10.0, 2)
    finally:
        client.close()

    return readings


# ── CSV file poller ───────────────────────────────────────────────────────────
def read_csv_logger(filepath: str) -> dict:
    """
    Read the latest row from a CSV that a data logger writes to continuously.
    The logger's CSV must have a datetime column and numeric sensor columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Logger CSV not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    # Find datetime column
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
    """
    Read from a data logger with a REST API.
    Expects a JSON response with numeric sensor fields.
    Adjust the field mapping below to match your logger's API.
    """
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


# ── Append to rolling CSV ─────────────────────────────────────────────────────
def log_reading(readings: dict, timestamp: pd.Timestamp):
    row = {"datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S")}
    row.update(readings)
    new_row = pd.DataFrame([row])

    # Append to full history
    if os.path.exists(READINGS_FILE):
        history = pd.read_csv(READINGS_FILE)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history.to_csv(READINGS_FILE, index=False)

    # Write rolling 24-hour window
    history["datetime"] = pd.to_datetime(history["datetime"])
    last_24h = history.tail(24)
    last_24h.to_csv(LAST_24H_FILE, index=False)

    # Write current SoC for cron use
    with open(SOC_FILE, "w") as f:
        f.write(str(readings.get("bess_soc", 60.0)))

    return len(history)


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_status(readings: dict, timestamp: pd.Timestamp, total_rows: int):
    print(f"\n  Timestamp:   {timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Load:        {readings['load_kw']:.1f} kW")
    print(f"  Solar:       {readings['solar_kw']:.1f} kW")
    print(f"  BESS SoC:    {readings['bess_soc']:.1f}%")
    print(f"  BESS power:  {readings['bess_kw']:+.1f} kW  "
          f"({'charging' if readings['bess_kw'] > 0 else 'discharging' if readings['bess_kw'] < 0 else 'idle'})")
    print(f"  Diesel:      {readings['diesel_kw']:.1f} kW")
    print(f"  Grid freq:   {readings['grid_freq']:.2f} Hz")
    print(f"\n  Total logged rows: {total_rows}")
    print(f"  History file:      {READINGS_FILE}")
    print(f"  Last 24h file:     {LAST_24H_FILE}")

    if total_rows >= 24 * 7:
        weeks = total_rows // (24 * 7)
        print(f"\n  ✓  {weeks} week(s) of data — sufficient for retraining.")
        print(f"     Run: python mgil_train.py --data {READINGS_FILE}")
    elif total_rows >= 24:
        days = total_rows // 24
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
    parser.add_argument("--port",    default="/dev/ttyUSB0", help="Modbus serial port")
    parser.add_argument("--slave",   type=int, default=1,    help="Modbus slave ID")
    parser.add_argument("--file",    default=None,           help="Logger CSV file path")
    parser.add_argument("--url",     default=None,           help="Logger HTTP endpoint")
    args = parser.parse_args()

    now = pd.Timestamp.now().floor("1h")

    print("=" * 60)
    print(f"  MGIL Logger — {now.strftime('%Y-%m-%d %H:%M')}  [{args.source}]")
    print("=" * 60)

    try:
        if args.source == "simulated":
            readings = read_simulated()
        elif args.source == "modbus":
            print(f"\n  Reading Modbus RTU on {args.port} (slave {args.slave})...")
            readings = read_modbus(port=args.port, slave_id=args.slave)
        elif args.source == "csv":
            if not args.file:
                print("  Error: --file is required for csv source.")
                sys.exit(1)
            print(f"\n  Reading logger CSV: {args.file}")
            readings = read_csv_logger(args.file)
        elif args.source == "http":
            if not args.url:
                print("  Error: --url is required for http source.")
                sys.exit(1)
            print(f"\n  Reading HTTP endpoint: {args.url}")
            readings = read_http(args.url)

        total_rows = log_reading(readings, now)
        print_status(readings, now, total_rows)
        print()

    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("  Check your data source connection and try again.\n")
        sys.exit(1)