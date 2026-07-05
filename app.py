"""
SW27 Flood Forecast API.

Serves on-demand LSTM forecasts for Stage_m and Discharge_m3s, predicted by a
Functional-API model with SEPARATE output heads sharing one LSTM trunk.
IMPORTANT: model.predict() on this model returns a LIST of two arrays
([stage_batch, discharge_log_batch]), not one combined array — this is
different from the earlier single-Dense(2)-output model and is the most
common source of shape-mismatch crashes when this file gets out of sync
with whatever architecture was actually trained.

Discharge is trained and predicted in LOG-SPACE (np.log1p at training time)
to handle its right-skewed distribution — every discharge value coming out
of the model must be passed through np.expm1() before use.

Currently seeded from a static historical file (seed_data.csv) exported from
the training notebook — labeled to visitors as a scenario-forecasting demo,
not live river conditions, until real telemetry is wired in (see
load_seed_data() below for the one function that needs to change then).
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ------------------------------------------------------------------
# Load model + scalers + config + seed data once, at startup
# ------------------------------------------------------------------
with open('config.json') as f:
    CONFIG = json.load(f)

LOOKBACK = CONFIG['lookback']
FEATURE_COLS = CONFIG['feature_cols']
TARGET_COLS = CONFIG.get('target_cols', ['Stage_m', 'Discharge_log'])
LAT, LON = CONFIG['lat'], CONFIG['lon']
TIMEZONE = CONFIG['timezone']

TEST_RMSE_STAGE = CONFIG.get('test_rmse_stage', CONFIG.get('test_rmse', 0.05))
TEST_RMSE_DISCHARGE = CONFIG.get('test_rmse_discharge', 0.0)

# Whether the second target column needs np.expm1() to become real discharge units.
# Checked by name rather than hardcoded, so this file keeps working if a future
# export goes back to raw (non-log) discharge as the target.
DISCHARGE_IS_LOG = 'Discharge_log' in TARGET_COLS

model = load_model('lstm_model.keras')

with open('feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)


def load_seed_data() -> pd.DataFrame:
    """
    Source of 'recent history' for seeding forecasts.

    Currently: a static CSV exported once from the training notebook.
    To switch to live telemetry later: replace this function's body with a
    call to your sensor API/database, keeping the same return shape
    (a DataFrame indexed by naive datetime, with columns == FEATURE_COLS,
    where Discharge_m3s stays in REAL units — the log-transform only ever
    applies to the training target, never to feature inputs).
    """
    df = pd.read_csv('seed_data.csv', parse_dates=['Timestamp'], index_col='Timestamp')
    return df


SEED_DATA = load_seed_data()
IS_LIVE_FEED = False  # flip to True once load_seed_data() pulls real-time telemetry

app = FastAPI(title="SW27 Flood Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def predict_step(window_batch: np.ndarray) -> tuple[float, float]:
    """
    Runs one forward pass and returns (predicted_stage_m, predicted_discharge_m3s)
    in REAL units, handling both possible model output shapes:
      - list of two arrays (separate-heads Functional model, current architecture)
      - single (1, 2) array (older single-Dense(2) architecture)
    """
    raw_pred = model.predict(window_batch, verbose=0)

    if isinstance(raw_pred, (list, tuple)):
        # Separate-heads model: [stage_array(1,1), discharge_array(1,1)]
        predicted_scaled = np.array([raw_pred[0][0][0], raw_pred[1][0][0]])
    else:
        # Single combined-output model: array shape (1, 2)
        predicted_scaled = raw_pred[0]

    if np.isnan(predicted_scaled).any():
        raise ValueError("NaN in model output")

    inv = target_scaler.inverse_transform([predicted_scaled])[0]
    predicted_stage = float(inv[0])
    predicted_discharge = float(np.expm1(inv[1])) if DISCHARGE_IS_LOG else float(inv[1])
    return predicted_stage, predicted_discharge


@app.get("/health")
def health():
    return {
        "status": "ok",
        "is_live_feed": IS_LIVE_FEED,
        "seed_data_last_timestamp": str(SEED_DATA.index.max()),
        "target_cols": TARGET_COLS,
        "discharge_is_log": DISCHARGE_IS_LOG,
    }


@app.get("/forecast")
def forecast(
    start: str = Query(..., description="ISO timestamp, e.g. 2026-07-05T08:00:00, or 'latest'"),
    horizon_value: int = Query(5, ge=1, le=1000),
    horizon_unit: str = Query("hours", pattern="^(hours|days|weeks)$"),
):
    # ---- Resolve start time ----
    if start.lower() == "latest":
        start_time_naive = SEED_DATA.index.max()
    else:
        try:
            start_time_naive = pd.Timestamp(start)
        except Exception:
            raise HTTPException(400, f"Could not parse start time: {start}")

    if horizon_unit == "hours":
        total_minutes = horizon_value * 60
    elif horizon_unit == "days":
        total_minutes = horizon_value * 24 * 60
    else:
        total_minutes = horizon_value * 7 * 24 * 60
    total_steps = int(total_minutes / 15)

    # Guard against very long recursive horizons stalling the request (see earlier
    # performance note: each step is its own sequential model.predict() call)
    MAX_STEPS = 672  # 7 days at 15-min resolution
    if total_steps > MAX_STEPS:
        raise HTTPException(
            400,
            f"Requested horizon ({total_steps} steps) exceeds the maximum of {MAX_STEPS} "
            f"steps (~7 days) supported per request."
        )

    # ---- Build seed window ----
    seed_source = SEED_DATA[SEED_DATA.index <= start_time_naive]
    if len(seed_source) < LOOKBACK:
        raise HTTPException(
            400,
            f"Not enough history before {start_time_naive} — need {LOOKBACK} rows, "
            f"have {len(seed_source)}. Earliest usable start is "
            f"{SEED_DATA.index[LOOKBACK]}."
        )

    last_historical_data = seed_source[FEATURE_COLS].iloc[-LOOKBACK:].copy()
    data_gap_minutes = (start_time_naive - last_historical_data.index.max()).total_seconds() / 60

    if last_historical_data.isna().any().any():
        last_historical_data = last_historical_data.ffill().bfill()

    current_window_scaled = feature_scaler.transform(
        pd.DataFrame(last_historical_data, columns=FEATURE_COLS)
    )
    if np.isnan(current_window_scaled).any():
        raise HTTPException(500, "Seed window produced NaN after scaling — check seed data.")

    current_window_batch = np.expand_dims(current_window_scaled, axis=0)
    rain_6h = last_historical_data['Rain_cumsum_6h'].iloc[-1]
    rain_24h = last_historical_data['Rain_cumsum_24h'].iloc[-1]

    # ---- Live rainfall forecast (this part IS genuinely live) ----
    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
        f"&hourly=precipitation&timezone={TIMEZONE.replace('/', '%2F')}"
    )
    try:
        f_response = requests.get(forecast_url, timeout=10).json()
        rain_times = pd.to_datetime(f_response['hourly']['time'])
        rain_forecast_df = pd.DataFrame(
            {'Precipitation_mm': f_response['hourly']['precipitation']}, index=rain_times
        )
        rain_forecast_df['Precipitation_mm'] = (
            rain_forecast_df['Precipitation_mm'].ffill().bfill().fillna(0.0)
        )
        rain_15m = rain_forecast_df.resample('15min').ffill() / 4
    except Exception:
        rain_15m = pd.DataFrame(columns=['Precipitation_mm'])

    # ---- Recursive forecast loop ----
    records = []
    for step in range(1, total_steps + 1):
        pred_time = start_time_naive + pd.Timedelta(minutes=15 * step)

        current_rain = 0.0
        if pred_time in rain_15m.index:
            val = rain_15m.loc[pred_time, 'Precipitation_mm']
            if not pd.isna(val):
                current_rain = float(val)

        try:
            predicted_stage, predicted_discharge = predict_step(current_window_batch)
        except ValueError:
            raise HTTPException(500, f"Model produced NaN at step {step} ({pred_time}).")

        stage_unc = TEST_RMSE_STAGE * np.sqrt(step)
        discharge_unc = TEST_RMSE_DISCHARGE * np.sqrt(step)

        records.append({
            "time": pred_time.isoformat(),
            "predicted_stage_m": round(predicted_stage, 4),
            "stage_upper_m": round(predicted_stage + stage_unc, 4),
            "stage_lower_m": round(predicted_stage - stage_unc, 4),
            "predicted_discharge_m3s": round(predicted_discharge, 4),
            "discharge_upper_m3s": round(predicted_discharge + discharge_unc, 4),
            "discharge_lower_m3s": round(predicted_discharge - discharge_unc, 4),
            "forecasted_rain_mm_hr": round(current_rain * 4, 3),
        })

        rain_6h = rain_6h + current_rain - (rain_6h / 24)
        rain_24h = rain_24h + current_rain - (rain_24h / 96)

        # Feed real-unit values back into the FEATURE window (inputs are never
        # log-transformed — only the training target was)
        new_step = pd.DataFrame(
            [[predicted_stage, predicted_discharge, current_rain, rain_6h, rain_24h]],
            columns=FEATURE_COLS
        )
        new_step_scaled = feature_scaler.transform(new_step)
        updated_window = np.vstack([current_window_batch[0][1:], new_step_scaled])
        current_window_batch = np.expand_dims(updated_window, axis=0)

    recent_history = SEED_DATA[SEED_DATA.index <= start_time_naive][['Stage_m', 'Discharge_m3s']].iloc[-96:]

    return {
        "is_live_feed": IS_LIVE_FEED,
        "seed_data_gap_minutes": round(data_gap_minutes, 1),
        "forecast_start": start_time_naive.isoformat(),
        "horizon_value": horizon_value,
        "horizon_unit": horizon_unit,
        "recent_history": [
            {"time": t.isoformat(), "stage_m": round(row.Stage_m, 4), "discharge_m3s": round(row.Discharge_m3s, 4)}
            for t, row in recent_history.iterrows()
        ],
        "forecast": records,
    }
