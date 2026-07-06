"""
SW27 Flood Forecast API.

Serves on-demand LSTM forecasts for Stage_m and Discharge_m3s, predicted by a
Functional-API model with SEPARATE output heads sharing one LSTM trunk.

IMPORTANT: this file rebuilds the model ARCHITECTURE directly in code and
loads only the trained WEIGHTS (model.weights.h5), rather than using
load_model() on the full .keras file. This sidesteps a class of error where
a model saved by a newer Keras version fails to deserialize on an older one
(e.g. "Unrecognized keyword arguments passed to Dense: {'quantization_config':
None}") — reconstructing the architecture directly avoids Keras needing to
interpret any saved config at all.

**If you change the model architecture in the notebook, mirror that change
in build_model() below, or the weights won't line up with the layers.**

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
import time
import warnings

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ------------------------------------------------------------------
# Load config + scalers + seed data
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
DISCHARGE_IS_LOG = 'Discharge_log' in TARGET_COLS

with open('feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)


# ------------------------------------------------------------------
# Rebuild the EXACT architecture from the notebook, then load weights only.
# Must match cell 14 of the training notebook layer-for-layer.
# ------------------------------------------------------------------
def build_model(lookback: int, n_features: int) -> Model:
    inputs = Input(shape=(lookback, n_features))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    shared = LSTM(32)(x)
    shared = Dropout(0.2)(shared)

    stage_head = Dense(16, activation='relu')(shared)
    stage_out = Dense(1, name='stage_output')(stage_head)

    discharge_head = Dense(16, activation='relu')(shared)
    discharge_out = Dense(1, name='discharge_output')(discharge_head)

    return Model(inputs=inputs, outputs=[stage_out, discharge_out])


model = build_model(LOOKBACK, len(FEATURE_COLS))
model.load_weights('model.weights.h5')

# One-time startup benchmark: prints raw per-step inference time to the logs.
# This tells us definitively whether slowness is inherent to running this
# model on this hardware, vs. something in the request-handling code around it.
_bench_start = time.time()
_dummy_input = np.zeros((1, LOOKBACK, len(FEATURE_COLS)), dtype=np.float32)
model(_dummy_input, training=False)  # first call (includes any lazy tracing/compile cost)
_first_call_seconds = time.time() - _bench_start

_bench_start = time.time()
for _ in range(5):
    model(_dummy_input, training=False)
_avg_call_seconds = (time.time() - _bench_start) / 5

print(f"[STARTUP BENCHMARK] First inference call: {_first_call_seconds:.2f}s | "
      f"Average of next 5 calls: {_avg_call_seconds:.3f}s/call", flush=True)


def load_seed_data() -> pd.DataFrame:
    """
    Source of 'recent history' for seeding forecasts.

    Currently: a static CSV exported once from the training notebook.
    Reads the FIRST column positionally as the datetime index, rather than by
    name — this avoids failures if master_df.index.name wasn't explicitly set
    before to_csv() in the notebook (a common source of "Missing column
    provided to 'parse_dates'" errors).

    To switch to live telemetry later: replace this function's body with a
    call to your sensor API/database, keeping the same return shape
    (a DataFrame indexed by naive datetime, with columns == FEATURE_COLS,
    where Discharge_m3s stays in REAL units — the log-transform only ever
    applies to the training target, never to feature inputs).
    """
    df = pd.read_csv('seed_data.csv', index_col=0, parse_dates=True)
    df.index.name = 'Timestamp'
    return df


SEED_DATA = load_seed_data()
IS_LIVE_FEED = False  # flip to True once load_seed_data() pulls real-time telemetry

# Simple TTL cache for the rainfall forecast — it doesn't meaningfully change
# minute-to-minute, so refetching it on every request just adds network
# latency for no benefit. Refreshed at most once every 10 minutes.
_rain_cache = {"data": None, "fetched_at": None}
RAIN_CACHE_TTL_SECONDS = 600

def get_rain_forecast() -> pd.DataFrame:
    now = pd.Timestamp.now()
    if (_rain_cache["data"] is not None and _rain_cache["fetched_at"] is not None
            and (now - _rain_cache["fetched_at"]).total_seconds() < RAIN_CACHE_TTL_SECONDS):
        return _rain_cache["data"]

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

    _rain_cache["data"] = rain_15m
    _rain_cache["fetched_at"] = now
    return rain_15m

app = FastAPI(title="SW27 Flood Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


import tensorflow as tf


def predict_step(window_batch: np.ndarray) -> tuple[float, float]:
    """Returns (predicted_stage_m, predicted_discharge_m3s) in REAL units.

    Uses a direct model call (model(x, training=False)) rather than
    model.predict() — .predict() carries per-call overhead (progress bar
    setup, callback handling) that's negligible for one call but adds up
    fast across a multi-step recursive forecast loop.
    """
    x = tf.convert_to_tensor(window_batch, dtype=tf.float32)
    stage_arr, discharge_arr = model(x, training=False)
    predicted_scaled = np.array([stage_arr.numpy()[0][0], discharge_arr.numpy()[0][0]])

    if np.isnan(predicted_scaled).any():
        raise ValueError("NaN in model output")

    inv = target_scaler.inverse_transform([predicted_scaled])[0]
    predicted_stage = float(inv[0])
    predicted_discharge = float(np.expm1(inv[1])) if DISCHARGE_IS_LOG else float(inv[1])
    return predicted_stage, predicted_discharge


def advance_one_step(window_batch, rain_6h, rain_24h, pred_time, rain_15m):
    """
    Runs one recursive step: predicts stage+discharge, updates the antecedent
    rainfall running totals, and slides the feature window forward by one row.
    Shared by both the bridge (real-data -> now) and the visitor-facing
    forecast loop, so they can never silently drift out of sync with each other.
    """
    current_rain = 0.0
    if pred_time in rain_15m.index:
        val = rain_15m.loc[pred_time, 'Precipitation_mm']
        if not pd.isna(val):
            current_rain = float(val)

    predicted_stage, predicted_discharge = predict_step(window_batch)

    rain_6h = rain_6h + current_rain - (rain_6h / 24)
    rain_24h = rain_24h + current_rain - (rain_24h / 96)

    new_step = pd.DataFrame(
        [[predicted_stage, predicted_discharge, current_rain, rain_6h, rain_24h]],
        columns=FEATURE_COLS
    )
    new_step_scaled = feature_scaler.transform(new_step)
    updated_window = np.vstack([window_batch[0][1:], new_step_scaled])
    updated_window_batch = np.expand_dims(updated_window, axis=0)

    return predicted_stage, predicted_discharge, current_rain, updated_window_batch, rain_6h, rain_24h


def build_initial_state_from_seed():
    """Builds the starting window/rain-totals from the last LOOKBACK rows of
    real seed data. This is the one true anchor point everything else —
    the bridge, and any forecast that starts within real data — builds from."""
    last_historical_data = SEED_DATA[FEATURE_COLS].iloc[-LOOKBACK:].copy()
    if last_historical_data.isna().any().any():
        last_historical_data = last_historical_data.ffill().bfill()

    window_scaled = feature_scaler.transform(pd.DataFrame(last_historical_data, columns=FEATURE_COLS))
    if np.isnan(window_scaled).any():
        raise HTTPException(500, "Seed window produced NaN after scaling — check seed data.")

    window_batch = np.expand_dims(window_scaled, axis=0)
    rain_6h = last_historical_data['Rain_cumsum_6h'].iloc[-1]
    rain_24h = last_historical_data['Rain_cumsum_24h'].iloc[-1]
    return window_batch, rain_6h, rain_24h, last_historical_data.index.max()


# ------------------------------------------------------------------
# Bridge cache: fills the gap between the last real data point and "now"
# using the model's own predictions, so visitors never see a dead gap on
# the chart. Computed incrementally — later requests only extend forward
# from wherever the cache already reached, instead of recomputing from
# scratch every time (recomputing a multi-day bridge on every request would
# be needlessly slow once dozens of visitors are hitting the same gap).
# ------------------------------------------------------------------
_bridge_cache = {
    "window_batch": None,
    "rain_6h": None,
    "rain_24h": None,
    "last_time": None,      # last timestamp the bridge has reached
    "records": [],          # list of dicts: time, stage, discharge, rain — bridged (synthetic) history
}
MAX_BRIDGE_STEPS_PER_CALL = 8  # ~2 hours — small safety top-up only; /advance-bridge (external cron) does the real work


import threading

_bridge_lock = threading.Lock()


def ensure_bridge_up_to(target_time: pd.Timestamp, rain_15m: pd.DataFrame):
    with _bridge_lock:
        if _bridge_cache["window_batch"] is None:
            window_batch, rain_6h, rain_24h, last_time = build_initial_state_from_seed()
            _bridge_cache.update({
                "window_batch": window_batch, "rain_6h": rain_6h,
                "rain_24h": rain_24h, "last_time": last_time, "records": [],
            })

        steps_needed = int((target_time - _bridge_cache["last_time"]).total_seconds() / 900)
        steps_needed = min(steps_needed, MAX_BRIDGE_STEPS_PER_CALL)

        for _ in range(max(steps_needed, 0)):
            next_time = _bridge_cache["last_time"] + pd.Timedelta(minutes=15)
            stage, discharge, rain, window_batch, rain_6h, rain_24h = advance_one_step(
                _bridge_cache["window_batch"], _bridge_cache["rain_6h"], _bridge_cache["rain_24h"],
                next_time, rain_15m
            )
            _bridge_cache["window_batch"] = window_batch
            _bridge_cache["rain_6h"] = rain_6h
            _bridge_cache["rain_24h"] = rain_24h
            _bridge_cache["last_time"] = next_time
            _bridge_cache["records"].append({
                "time": next_time, "stage_m": stage, "discharge_m3s": discharge, "rain_mm_hr": rain * 4
            })
            if len(_bridge_cache["records"]) > 200:
                _bridge_cache["records"] = _bridge_cache["records"][-200:]

        return dict(_bridge_cache)  # shallow copy so callers don't race with ongoing background updates


# The background thread approach (continuously advancing the bridge inside
# this same process) was removed: on constrained free-tier CPU, a Python
# thread doing repeated TensorFlow inference can starve the main request-
# serving thread of CPU time via the GIL, even while "backing off" between
# batches — this made the API unreachable for minutes at a time despite
# Render reporting the container as "Live" (that status only confirms the
# process is running and bound to a port, not that it's responsive).
#
# Instead, /advance-bridge below does one small, bounded chunk of catch-up
# work per call, and is meant to be triggered externally on a schedule (see
# the GitHub Actions workflow) — so the heavy lifting happens in short,
# controlled bursts rather than a thread perpetually competing for CPU.
# Instead, /advance-bridge below does one small, bounded chunk of catch-up
# work per call, and is meant to be triggered externally on a schedule (see
# the GitHub Actions workflow) — so the heavy lifting happens in short,
# controlled bursts rather than a thread perpetually competing for CPU.


@app.get("/advance-bridge")
def advance_bridge():
    """
    Advances the bridge by a small, bounded amount (~10 hours max per call)
    and returns immediately. Meant to be hit periodically by an external
    scheduler (GitHub Actions cron), NOT by visitor traffic — this keeps the
    heavy recursive computation in short controlled bursts instead of a
    continuously-running thread competing with real requests for CPU.
    """
    rain_15m = get_rain_forecast()
    target = pd.Timestamp.now().floor('15min')
    bridge = ensure_bridge_up_to(target, rain_15m)
    minutes_behind = round((pd.Timestamp.now() - bridge["last_time"]).total_seconds() / 60, 1)
    return {
        "bridge_caught_up_to": str(bridge["last_time"]),
        "minutes_behind_now": minutes_behind,
    }


@app.get("/health")
def health():
    bridge_last_time = _bridge_cache["last_time"]
    return {
        "status": "ok",
        "is_live_feed": IS_LIVE_FEED,
        "seed_data_last_timestamp": str(SEED_DATA.index.max()),
        "target_cols": TARGET_COLS,
        "discharge_is_log": DISCHARGE_IS_LOG,
        "bridge_caught_up_to": str(bridge_last_time) if bridge_last_time is not None else None,
        "bridge_minutes_behind_now": (
            round((pd.Timestamp.now() - bridge_last_time).total_seconds() / 60, 1)
            if bridge_last_time is not None else None
        ),
    }


@app.get("/forecast")
def forecast(
    start: str = Query(..., description="ISO timestamp, e.g. 2026-07-05T08:00:00, or 'latest'"),
    horizon_value: int = Query(5, ge=1, le=1000),
    horizon_unit: str = Query("hours", pattern="^(hours|days|weeks)$"),
):
    _req_start = time.time()
    print(f"[REQUEST] /forecast start={start} horizon={horizon_value}{horizon_unit}", flush=True)

    # ---- Resolve start time ----
    if start.lower() == "latest":
        start_time_naive = pd.Timestamp.now().floor('15min')
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

    MAX_STEPS = 288  # 3 days at 15-min resolution — kept deliberately bounded so a single
                      # request can't run long enough to risk a proxy/gateway timeout on free hosting
    if total_steps > MAX_STEPS:
        raise HTTPException(
            400,
            f"Requested horizon ({total_steps} steps) exceeds the maximum of {MAX_STEPS} "
            f"steps (~7 days) supported per request."
        )

    _t = time.time()
    rain_15m = get_rain_forecast()
    print(f"[TIMING] get_rain_forecast: {time.time() - _t:.2f}s", flush=True)

    real_data_end = SEED_DATA.index.max()
    data_gap_minutes = max((start_time_naive - real_data_end).total_seconds() / 60, 0)
    requested_start_time = start_time_naive  # keep the original ask for reporting, in case we fall back
    still_catching_up = False

    if start_time_naive <= real_data_end:
        # Requested start falls within real data — no bridging needed, seed directly from it.
        seed_source = SEED_DATA[SEED_DATA.index <= start_time_naive]
        if len(seed_source) < LOOKBACK:
            raise HTTPException(
                400,
                f"Not enough history before {start_time_naive} — need {LOOKBACK} rows, "
                f"have {len(seed_source)}. Earliest usable start is {SEED_DATA.index[LOOKBACK]}."
            )
        last_historical_data = seed_source[FEATURE_COLS].iloc[-LOOKBACK:].copy()
        if last_historical_data.isna().any().any():
            last_historical_data = last_historical_data.ffill().bfill()
        window_scaled = feature_scaler.transform(pd.DataFrame(last_historical_data, columns=FEATURE_COLS))
        current_window_batch = np.expand_dims(window_scaled, axis=0)
        rain_6h = last_historical_data['Rain_cumsum_6h'].iloc[-1]
        rain_24h = last_historical_data['Rain_cumsum_24h'].iloc[-1]
        recent_context = [
            {"time": t.isoformat(), "stage_m": round(row.Stage_m, 4), "discharge_m3s": round(row.Discharge_m3s, 4)}
            for t, row in seed_source[['Stage_m', 'Discharge_m3s']].iloc[-32:].iterrows()
        ]
    else:
        # Requested start is beyond real data — use the bridge (built mostly by
        # the background warmup worker; this only tops up any small residual gap).
        _t = time.time()
        bridge = ensure_bridge_up_to(start_time_naive, rain_15m)
        print(f"[TIMING] ensure_bridge_up_to: {time.time() - _t:.2f}s", flush=True)
        current_window_batch = bridge["window_batch"]
        rain_6h = bridge["rain_6h"]
        rain_24h = bridge["rain_24h"]

        # If the background worker hasn't caught all the way up yet (e.g. right
        # after a fresh deploy), forecast from wherever the bridge actually is
        # rather than silently pretending we reached the requested start.
        actual_anchor_time = bridge["last_time"]
        if actual_anchor_time < start_time_naive:
            still_catching_up = True
            start_time_naive = actual_anchor_time

        recent_context = [
            {"time": r["time"].isoformat(), "stage_m": round(r["stage_m"], 4), "discharge_m3s": round(r["discharge_m3s"], 4)}
            for r in bridge["records"][-32:]
        ]

    # ---- Recursive forecast loop for the visitor's actual requested horizon ----
    _t = time.time()
    records = []
    for step in range(1, total_steps + 1):
        pred_time = start_time_naive + pd.Timedelta(minutes=15 * step)
        try:
            stage, discharge, rain, current_window_batch, rain_6h, rain_24h = advance_one_step(
                current_window_batch, rain_6h, rain_24h, pred_time, rain_15m
            )
        except ValueError:
            raise HTTPException(500, f"Model produced NaN at step {step} ({pred_time}).")

        stage_unc = TEST_RMSE_STAGE * np.sqrt(step)
        discharge_unc = TEST_RMSE_DISCHARGE * np.sqrt(step)

        records.append({
            "time": pred_time.isoformat(),
            "predicted_stage_m": round(stage, 4),
            "stage_upper_m": round(stage + stage_unc, 4),
            "stage_lower_m": round(max(stage - stage_unc, 0), 4),
            "predicted_discharge_m3s": round(discharge, 4),
            "discharge_upper_m3s": round(discharge + discharge_unc, 4),
            "discharge_lower_m3s": round(max(discharge - discharge_unc, 0), 4),  # discharge can't be negative
            "forecasted_rain_mm_hr": round(rain * 4, 3),
        })

    _loop_seconds = time.time() - _t
    print(f"[TIMING] forecast loop ({total_steps} steps): {_loop_seconds:.2f}s "
          f"({_loop_seconds/max(total_steps,1)*1000:.0f}ms/step)", flush=True)
    print(f"[TIMING] TOTAL request time: {time.time() - _req_start:.2f}s", flush=True)

    return {
        "is_live_feed": IS_LIVE_FEED,
        "seed_data_gap_minutes": round(data_gap_minutes, 1),
        "bridged": requested_start_time > real_data_end,
        "still_catching_up": still_catching_up,
        "requested_start": requested_start_time.isoformat(),
        "forecast_start": start_time_naive.isoformat(),
        "horizon_value": horizon_value,
        "horizon_unit": horizon_unit,
        "recent_history": recent_context,
        "forecast": records,
    }
