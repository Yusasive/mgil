# MGIL — Mini-Grid Intelligence Layer

**Predictive Load Forecasting and Smart Energy Management for Solar Hybrid Mini-Grids**

> Built for Nigeria's DARES programme — 1,350 mini-grids serving rural communities across 36 states.

---

## The Problem

Diesel generators account for 60-80% of operating costs at most Nigerian mini-grids. The root cause is not the price of fuel — it is the absence of intelligent dispatch. Without a forecast of what the load will be in the next 24 hours, operators cannot plan when to charge the battery, when to rely on solar, and when diesel is genuinely unavoidable. The result is preventable diesel runtime, premature battery degradation, and unnecessary cost passed on to consumers who can least afford it.

## What MGIL Does

MGIL is a software system that runs on a $35 Raspberry Pi installed at the mini-grid site. Every morning it:

1. Reads the last 24 hours of sensor data from the site's data logger
2. Produces a **24-hour load forecast** using an ensemble of LSTM and Random Forest models
3. Runs the **SEMS dispatcher** — a 6-hour look-ahead controller that plans each hour's energy source
4. Writes a dispatch plan and daily report for the site operator

No internet connection required. No cloud subscription. The intelligence runs at the edge, on the mini-grid itself.

**Validated results (paper):** 7.3% MAPE on load forecasting, 34.2% reduction in diesel runtime versus baseline dispatch.

---

## Project Structure

```
mgil/
├── mgil_log.py          ← runs every hour: reads sensors, builds data history
├── mgil_train.py        ← run once on install, then monthly: trains ML models
├── mgil_predict.py      ← runs every morning: 24-hour forecast + dispatch plan
├── requirements.txt     ← Python dependencies
├── README.md            ← this file
│
├── data/                ← created automatically by mgil_log.py
│   ├── site_readings.csv    full sensor history (grows daily)
│   ├── last_24h.csv         rolling 24-hour window for forecasting
│   └── current_soc.txt      latest BESS state of charge
│
├── models/              ← created automatically by mgil_train.py
│   ├── lstm_model.keras
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── meta.json
│   └── training_report.txt
│
├── forecast/            ← created automatically by mgil_predict.py
│   ├── forecast_YYYY-MM-DD.csv
│   ├── dispatch_YYYY-MM-DD.csv
│   └── report_YYYY-MM-DD.txt
│
└── logs/                ← cron job output
    └── mgil.log
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Computer | Raspberry Pi 4B (4 GB RAM) |
| Storage | 32 GB microSD card |
| Power | 5V 3A USB-C supply |
| Data link | RS-485 to USB adapter (for Modbus data loggers) or Ethernet |
| Cost | ~$35 USD total |

---

## Installation

### 1. Flash the SD card

Download Raspberry Pi OS Lite (64-bit) and flash it to the SD card. Enable SSH during setup.

### 2. Copy the MGIL package to the Pi

```bash
scp -r mgil/ pi@<pi-ip-address>:/home/pi/mgil
```

Or copy via USB stick if the site has no network.

### 3. Create the Python virtual environment

```bash
cd /home/pi/mgil
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This takes 10-15 minutes on a Raspberry Pi 4B. Run it once and leave it.

### 4. Train the models

**Option A: No site data yet (first installation)**
```bash
python mgil_train.py
```
Trains on a synthetic DARES baseline profile. Good enough to start; accuracy improves once real data is collected.

**Option B: Real site data available**
```bash
python mgil_train.py --data data/site_readings.csv
```
Expects columns: `datetime`, `load_kw`, `solar_kw`, `bess_soc`. Column names are flexible — the script accepts common variants automatically.

Training takes about 5 minutes. Check `models/training_report.txt` for MAPE accuracy.

### 5. Test the full pipeline

```bash
# Test data logging (simulated sensors)
python mgil_log.py

# Test forecasting
python mgil_predict.py --soc 65.0
```

You should see a 24-hour hourly dispatch table and CSV files saved to `forecast/`.

### 6. Schedule automatic operation (cron)

```bash
crontab -e
```

Add these three lines:

```
# Log sensor readings every hour
0 * * * * /home/pi/mgil/venv/bin/python /home/pi/mgil/mgil_log.py --source modbus >> /home/pi/mgil/logs/mgil.log 2>&1

# Run daily forecast every morning at 6:00 AM
0 6 * * * /home/pi/mgil/venv/bin/python /home/pi/mgil/mgil_predict.py --input /home/pi/mgil/data/last_24h.csv --soc $(cat /home/pi/mgil/data/current_soc.txt) >> /home/pi/mgil/logs/mgil.log 2>&1

# Retrain models every Sunday at 2:00 AM
0 2 * * 0 /home/pi/mgil/venv/bin/python /home/pi/mgil/mgil_train.py --data /home/pi/mgil/data/site_readings.csv >> /home/pi/mgil/logs/mgil.log 2>&1
```

From this point the system runs fully autonomously.

---

## Data Collection (mgil_log.py)

The logger reads one set of sensor values per hour and appends them to `data/site_readings.csv`. It supports four data sources:

| Source | Flag | When to use |
|--------|------|-------------|
| Simulated | `--source simulated` | Testing, demo, no hardware connected |
| Modbus RTU | `--source modbus` | Most solar inverters and charge controllers (Victron, SMA, Growatt) |
| CSV polling | `--source csv --file /path/to/logger.csv` | Loggers that write their own output file |
| HTTP/REST | `--source http --url http://192.168.1.x/api` | Loggers with a web interface |

After 7 days of logging, the script will confirm you have enough data to retrain. After 30 days, model accuracy on your specific site's load pattern will significantly exceed the synthetic baseline.

---

## Daily Operation

| File | Description |
|------|-------------|
| `forecast/forecast_YYYY-MM-DD.csv` | 24-hour load and solar forecast |
| `forecast/dispatch_YYYY-MM-DD.csv` | Full SEMS dispatch plan, one row per hour |
| `forecast/report_YYYY-MM-DD.txt` | Human-readable summary for the site operator |
| `logs/mgil.log` | Cron job output — check here if something goes wrong |

### Dispatch CSV columns

| Column | Meaning |
|--------|---------|
| `state` | `PV_SURPLUS` / `BESS_COVER` / `DIESEL_REQUIRED` / `PRE_CHARGE` |
| `forecast_kw` | Predicted load demand for that hour |
| `solar_kw` | Predicted solar generation |
| `bess_to_load` | Power discharged from BESS to serve load |
| `pv_to_bess` | Excess PV power routed to charge BESS |
| `diesel_kw` | Diesel generator output required |
| `bess_soc_end` | BESS state of charge at end of hour (%) |

---

## SEMS Dispatch States

| State | Condition | Action |
|-------|-----------|--------|
| `PV_SURPLUS` | Solar exceeds load | PV serves load; surplus charges BESS |
| `BESS_COVER` | Solar below load, BESS has capacity | BESS covers the deficit, no diesel |
| `DIESEL_REQUIRED` | Solar and BESS both insufficient | Diesel generator starts |
| `PRE_CHARGE` | Mid-day, SoC below 70%, evening peak ahead | Aggressively charge BESS from excess PV |

---

## Model Architecture

| Model | Architecture | Input features |
|-------|-------------|---------------|
| LSTM | 2 layers (64 + 32 units), Dropout 0.2, EarlyStopping | 24-hour scaled load sequence |
| Random Forest | 200 trees, max depth 15, min samples leaf 2 | Hour, day, month, seasonal flags (Harmattan, harvest, market day), 6 lag features |
| Ensemble | 40% LSTM + 60% Random Forest | Combined weighted prediction |

Typical accuracy on DARES-profile sites: **MAPE < 8%**

---

## Retraining

Retrain every 3-6 months as the site accumulates real operating data:

```bash
source venv/bin/activate
python mgil_train.py --data data/site_readings.csv
```

The old models are replaced in place. The Sunday cron job does this automatically once weekly.

---

## Troubleshooting

**"Models not found. Run mgil_train.py first."**
The training script has not been run yet, or was run from a different directory. Run `python mgil_train.py` from inside the `mgil/` folder.

**"No space left on device"**
The `forecast/` and `logs/` directories grow over time. Delete files older than 90 days:
```bash
find /home/pi/mgil/forecast -mtime +90 -delete
find /home/pi/mgil/logs -mtime +90 -delete
```

**Forecast accuracy is poor after installation**
Expected — the synthetic baseline does not know your site. Allow 7-14 days of logging, then retrain: `python mgil_train.py --data data/site_readings.csv`

**Cron job is not running**
Check the log: `cat /home/pi/mgil/logs/mgil.log`. Verify the Python path is correct with: `which python` inside the activated venv.

**Modbus connection fails**
Confirm the RS-485 adapter is detected: `ls /dev/ttyUSB*`. Check the slave ID and baud rate in `mgil_log.py` match your data logger's manual.

---

## Citation

If you use MGIL in your research or deployment work, please cite:

> Abdullahi Temidayo, Y. (2026). *A Machine Learning Framework for Predictive Load Forecasting and Smart Energy Management in Solar Hybrid Mini-Grids.* Russian-African Forum of Young Scientists, Saint Petersburg.

---

*MGIL v1.0 — Yusuf Abdullahi Temidayo*  
*Built for the DARES programme, Nigeria*