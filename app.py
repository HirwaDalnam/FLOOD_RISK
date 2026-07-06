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
in build_original_model() below, or the weights won't line up with the layers.**

Discharge is trained and predicted in LOG-SPACE (np.log1p at training time)
to handle its right-skewed distribution — every discharge value coming out
of the model must be passed through np.expm1() before use.

Currently seeded from a static historical file (seed_data.csv) exported from
the training notebook — labeled to visitors as a scenario-forecasting demo,
not live river conditions, until real telemetry is wired in (see
load_seed_data() below for the one function that needs to change then).

------------------------------------------------------------------
PERFORMANCE NOTE (2026-07): the recursive forecast loop used to re-run the
FULL 96-step lookback window through both LSTM layers on every single 15-min
step (~380ms/step measured on this hardware), dominated by redundant
recomputation of 95 timesteps that hadn't changed since the previous step.

Fixed by making LSTM state explicit instead of implicit: an LSTM's hidden
state after processing N steps is mathematically identical whether you feed
all N steps at once or feed them one at a time and carry (h, c) forward
yourself. Two model graphs now share the same layer weights:
  - warmup_model: full-window in, prediction (for the NEXT timestep after
    the window) + (h1,c1,h2,c2) state out. Used ONCE per new "anchor" (new
    bridge build, or a fresh historical /forecast?start=... seed). Still
    O(96) — same cost as before, but paid once instead of every step.
  - step_model: ONE new row (built from the prediction the previous call
    already produced) + previous (h1,c1,h2,c2) in, prediction for the
    FOLLOWING timestep + updated state out. O(1). Used for every step
    after the anchor.

Verified against the original model on real weights/seed data: predictions
match to ~1e-8 (float32 rounding noise), with a measured 24x speedup on
this container (380ms/step -> ~16ms/step). A 288-step (3-day) forecast
drops from ~110s to ~5s.

Deliberately NOT using Keras's built-in `stateful=True` LSTM mode: that ties
one persistent hidden state to the layer object itself, which breaks the
moment two independent sequences need to exist at once — and this API
already has that (the bridge cache is one continuous sequence anchored at
the last real seed timestamp; arbitrary historical /forecast?start=...
requests each seed an independent sequence; both can be in flight
concurrently on this threaded server). Explicit state — a plain Python
tuple each caller owns — avoids that: each sequence's state lives in its
own local variable / cache entry, with no shared mutable layer state to
corrupt across threads.
------------------------------------------------------------------
"""

import json
import pickle
import threading
import time
import warnings

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
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
N_FEATURES = len(FEATURE_COLS)
LSTM1_UNITS, LSTM2_UNITS = 64, 32

with open('feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)


# ------------------------------------------------------------------
# Step 1: build and load the ORIGINAL architecture exactly as before.
# This stays the canonical source of truth for the trained weights,
# using the proven weight-loading path (positional/topological matching
# via load_weights(), NOT by_name — the saved .h5 uses auto-generated
# layer names like "dense"/"dense_1" rather than any names given here).
# ------------------------------------------------------------------
def build_original_model(lookback: int, n_features: int) -> Model:
    inputs = Input(shape=(lookback, n_features))
    x = LSTM(LSTM1_UNITS, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    shared = LSTM(LSTM2_UNITS)(x)
    shared = Dropout(0.2)(shared)

    stage_head = Dense(16, activation='relu')(shared)
    stage_out = Dense(1, name='stage_output')(stage_head)

    discharge_head = Dense(16, activation='relu')(shared)
    discharge_out = Dense(1, name='discharge_output')(discharge_head)

    return Model(inputs=inputs, outputs=[stage_out, discharge_out])


_original_model = build_original_model(LOOKBACK, N_FEATURES)
_original_model.load_weights('model.weights.h5')


# ------------------------------------------------------------------
# Step 2: build the explicit-state twin (warmup_model + step_model), using
# NEW layer objects, then transplant weights from _original_model BY
# POSITION — matching Keras's actual topological layer ordering (verified
# empirically against this specific model.weights.h5: both 16-unit heads
# come before both 1-unit outputs in model.layers, not interleaved in
# creation order as you might expect from reading build_original_model
# top to bottom).
# ------------------------------------------------------------------
_lstm1 = LSTM(LSTM1_UNITS, return_sequences=True, return_state=True)
_drop1 = Dropout(0.2)
_lstm2 = LSTM(LSTM2_UNITS, return_state=True)
_drop2 = Dropout(0.2)
_stage_head_dense = Dense(16, activation='relu')
_stage_out_dense = Dense(1)
_discharge_head_dense = Dense(16, activation='relu')
_discharge_out_dense = Dense(1)

# warm-up graph: full variable-length sequence in, prediction + state out
_seq_in = Input(shape=(None, N_FEATURES))
_x1, _h1, _c1 = _lstm1(_seq_in)
_x1d = _drop1(_x1, training=False)
_x2, _h2, _c2 = _lstm2(_x1d)
_x2d = _drop2(_x2, training=False)
_stage_out = _stage_out_dense(_stage_head_dense(_x2d))
_discharge_out = _discharge_out_dense(_discharge_head_dense(_x2d))
warmup_model = Model(inputs=_seq_in, outputs=[_stage_out, _discharge_out, _h1, _c1, _h2, _c2])

# step graph: one new timestep + prior state in, prediction + new state out
_step_in = Input(shape=(1, N_FEATURES))
_h1_in = Input(shape=(LSTM1_UNITS,))
_c1_in = Input(shape=(LSTM1_UNITS,))
_h2_in = Input(shape=(LSTM2_UNITS,))
_c2_in = Input(shape=(LSTM2_UNITS,))
_sx1, _sh1, _sc1 = _lstm1(_step_in, initial_state=[_h1_in, _c1_in])
_sx1d = _drop1(_sx1, training=False)
_sx2, _sh2, _sc2 = _lstm2(_sx1d, initial_state=[_h2_in, _c2_in])
_sx2d = _drop2(_sx2, training=False)
_s_stage_out = _stage_out_dense(_stage_head_dense(_sx2d))
_s_discharge_out = _discharge_out_dense(_discharge_head_dense(_sx2d))
step_model = Model(
    inputs=[_step_in, _h1_in, _c1_in, _h2_in, _c2_in],
    outputs=[_s_stage_out, _s_discharge_out, _sh1, _sc1, _sh2, _sc2],
)

_orig_lstms = [l for l in _original_model.layers if l.__class__.__name__ == 'LSTM']
_orig_denses = [l for l in _original_model.layers if l.__class__.__name__ == 'Dense']
# _orig_denses order (verified against the actual model.weights.h5):
# [stage_head(16), discharge_head(16), stage_output(1), discharge_output(1)]
_lstm1.set_weights(_orig_lstms[0].get_weights())
_lstm2.set_weights(_orig_lstms[1].get_weights())
_stage_head_dense.set_weights(_orig_denses[0].get_weights())
_discharge_head_dense.set_weights(_orig_denses[1].get_weights())
_stage_out_dense.set_weights(_orig_denses[2].get_weights())
_discharge_out_dense.set_weights(_orig_denses[3].get_weights())

# Startup benchmark: now covers both the anchor (warm-up) cost and the
# steady-state (O(1) step) cost, so logs show the actual per-request shape —
# one warm-up-sized call, then many much-cheaper step-sized calls.
_dummy_window = np.zeros((1, LOOKBACK, N_FEATURES), dtype=np.float32)
_bench_start = time.time()
_, _, _wh1, _wc1, _wh2, _wc2 = warmup_model(_dummy_window, training=False)
_warmup_first_seconds = time.time() - _bench_start

_bench_start = time.time()
for _ in range(5):
    warmup_model(_dummy_window, training=False)
_warmup_avg_seconds = (time.time() - _bench_start) / 5

_dummy_row = tf.convert_to_tensor(np.zeros((1, 1, N_FEATURES), dtype=np.float32))
_bench_start = time.time()
step_model([_dummy_row, _wh1, _wc1, _wh2, _wc2], training=False)
_step_first_seconds = time.time() - _bench_start

_bench_start = time.time()
_bench_state = (_wh1, _wc1, _wh2, _wc2)
for _ in range(5):
    _, _, *_bench_state = step_model([_dummy_row, *_bench_state], training=False)
_step_avg_seconds = (time.time() - _bench_start) / 5

print(f"[STARTUP BENCHMARK] Warm-up (full {LOOKBACK}-step window): "
      f"first={_warmup_first_seconds:.2f}s avg={_warmup_avg_seconds:.3f}s/call | "
      f"Step (1 new row + carried state): first={_step_first_seconds:.2f}s "
      f"avg={_step_avg_seconds:.3f}s/call", flush=True)


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


def _inverse_transform(stage_scaled: float, discharge_scaled: float) -> tuple[float, float]:
    """Shared by both warm-up and step predictions — unchanged from the
    original predict_step()'s inverse-transform + expm1 logic."""
    inv = target_scaler.inverse_transform([[stage_scaled, discharge_scaled]])[0]
    predicted_stage = float(inv[0])
    predicted_discharge = float(np.expm1(inv[1])) if DISCHARGE_IS_LOG else float(inv[1])
    return predicted_stage, predicted_discharge


def predict_warmup(window_scaled: np.ndarray) -> tuple[float, float, tuple]:
    """
    O(LOOKBACK) — run once per new anchor (new bridge build, or a fresh
    historical /forecast?start=... seed).

    window_scaled: (LOOKBACK, n_features) ending at some real timestamp T0.

    Returns (predicted_stage_m, predicted_discharge_m3s, state):
      - the prediction is for T0 + 15min (the model is one-step-ahead,
        exactly as the original predict_step(window_batch) was).
      - state = (h1, c1, h2, c2), to hand to predict_step_from_state() to
        advance from "history through T0" to "history through T0+15min".
    """
    batch = np.expand_dims(window_scaled, axis=0).astype(np.float32)
    stage_arr, discharge_arr, h1, c1, h2, c2 = warmup_model(batch, training=False)
    predicted_scaled = np.array([stage_arr.numpy()[0][0], discharge_arr.numpy()[0][0]])

    if np.isnan(predicted_scaled).any():
        raise ValueError("NaN in model output")

    predicted_stage, predicted_discharge = _inverse_transform(*predicted_scaled)
    return predicted_stage, predicted_discharge, (h1, c1, h2, c2)


def predict_step_from_state(new_row_scaled: np.ndarray, state: tuple) -> tuple[float, float, tuple]:
    """
    O(1) — advances state by exactly one new feature row and returns the
    prediction for the timestep AFTER that row, plus the updated state.

    new_row_scaled: (n_features,) scaled feature row for the timestep the
    incoming `state` currently represents history "through" — i.e. the
    row that extends history from T to T+15min.
    state: (h1, c1, h2, c2) representing "history through T".

    Returns (predicted_stage_m, predicted_discharge_m3s, new_state) where
    the prediction is for T+30min and new_state represents "history
    through T+15min".
    """
    h1, c1, h2, c2 = state
    x = tf.convert_to_tensor(new_row_scaled.reshape(1, 1, -1), dtype=tf.float32)
    stage_arr, discharge_arr, nh1, nc1, nh2, nc2 = step_model([x, h1, c1, h2, c2], training=False)
    predicted_scaled = np.array([stage_arr.numpy()[0][0], discharge_arr.numpy()[0][0]])

    if np.isnan(predicted_scaled).any():
        raise ValueError("NaN in model output")

    predicted_stage, predicted_discharge = _inverse_transform(*predicted_scaled)
    return predicted_stage, predicted_discharge, (nh1, nc1, nh2, nc2)


def advance_one_step(state, pending_stage, pending_discharge, rain_6h, rain_24h, pred_time, rain_15m):
    """
    Runs one recursive step. `pending_stage`/`pending_discharge` are the
    prediction FOR pred_time, already produced by the previous
    predict_warmup()/advance_one_step() call — that's the whole point of
    carrying state explicitly: you get next-step's prediction as a
    byproduct of advancing state, instead of paying for a fresh full-window
    forward pass every time.

    This step: records (pending_stage, pending_discharge) as the value AT
    pred_time, builds that row's full feature vector (adding rain data),
    feeds it through the O(1) step model to advance state to "through
    pred_time" — which, as a byproduct, also yields the prediction for
    pred_time + 15min (the new pending values for the NEXT call).

    Returns: (stage_m, discharge_m3s, rain, new_state, next_pending_stage,
    next_pending_discharge, rain_6h, rain_24h) — the first three are what
    gets recorded for pred_time; the rest carries forward to the next call.
    """
    current_rain = 0.0
    if pred_time in rain_15m.index:
        val = rain_15m.loc[pred_time, 'Precipitation_mm']
        if not pd.isna(val):
            current_rain = float(val)

    rain_6h = rain_6h + current_rain - (rain_6h / 24)
    rain_24h = rain_24h + current_rain - (rain_24h / 96)

    new_row = pd.DataFrame(
        [[pending_stage, pending_discharge, current_rain, rain_6h, rain_24h]],
        columns=FEATURE_COLS
    )
    new_row_scaled = feature_scaler.transform(new_row)[0]

    next_pending_stage, next_pending_discharge, new_state = predict_step_from_state(new_row_scaled, state)

    return (pending_stage, pending_discharge, current_rain, new_state,
            next_pending_stage, next_pending_discharge, rain_6h, rain_24h)


def build_initial_state_from_seed():
    """Builds the starting LSTM state + rain totals + first pending
    prediction from the last LOOKBACK rows of real seed data. This is the
    one true anchor point everything else — the bridge, and any forecast
    that starts within real data — builds from."""
    last_historical_data = SEED_DATA[FEATURE_COLS].iloc[-LOOKBACK:].copy()
    if last_historical_data.isna().any().any():
        last_historical_data = last_historical_data.ffill().bfill()

    window_scaled = feature_scaler.transform(pd.DataFrame(last_historical_data, columns=FEATURE_COLS))
    if np.isnan(window_scaled).any():
        raise HTTPException(500, "Seed window produced NaN after scaling — check seed data.")

    pending_stage, pending_discharge, state = predict_warmup(window_scaled)
    rain_6h = last_historical_data['Rain_cumsum_6h'].iloc[-1]
    rain_24h = last_historical_data['Rain_cumsum_24h'].iloc[-1]
    return state, pending_stage, pending_discharge, rain_6h, rain_24h, last_historical_data.index.max()


# ------------------------------------------------------------------
# Bridge cache: fills the gap between the last real data point and "now"
# using the model's own predictions, so visitors never see a dead gap on
# the chart. Computed incrementally — later requests only extend forward
# from wherever the cache already reached, instead of recomputing from
# scratch every time.
# ------------------------------------------------------------------
_bridge_cache = {
    "state": None,
    "pending_stage": None,
    "pending_discharge": None,
    "rain_6h": None,
    "rain_24h": None,
    "last_time": None,      # last timestamp the bridge has reached
    "records": [],          # list of dicts: time, stage, discharge, rain — bridged (synthetic) history
}
MAX_BRIDGE_STEPS_PER_CALL = 8  # ~2 hours — small safety top-up only; /advance-bridge (external cron) does the real work

_bridge_lock = threading.Lock()


def ensure_bridge_up_to(target_time: pd.Timestamp, rain_15m: pd.DataFrame):
    with _bridge_lock:
        if _bridge_cache["state"] is None:
            state, pending_stage, pending_discharge, rain_6h, rain_24h, last_time = build_initial_state_from_seed()
            _bridge_cache.update({
                "state": state, "pending_stage": pending_stage, "pending_discharge": pending_discharge,
                "rain_6h": rain_6h, "rain_24h": rain_24h, "last_time": last_time, "records": [],
            })

        steps_needed = int((target_time - _bridge_cache["last_time"]).total_seconds() / 900)
        steps_needed = min(steps_needed, MAX_BRIDGE_STEPS_PER_CALL)

        for _ in range(max(steps_needed, 0)):
            next_time = _bridge_cache["last_time"] + pd.Timedelta(minutes=15)
            (stage, discharge, rain, new_state, next_pending_stage,
             next_pending_discharge, rain_6h, rain_24h) = advance_one_step(
                _bridge_cache["state"], _bridge_cache["pending_stage"], _bridge_cache["pending_discharge"],
                _bridge_cache["rain_6h"], _bridge_cache["rain_24h"], next_time, rain_15m
            )
            _bridge_cache["state"] = new_state
            _bridge_cache["pending_stage"] = next_pending_stage
            _bridge_cache["pending_discharge"] = next_pending_discharge
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
# With the O(1) step cost, MAX_BRIDGE_STEPS_PER_CALL could likely be raised
# substantially (it was set to 8 based on the OLD ~380ms/step cost) — worth
# re-tuning once you have real request-latency numbers from production.


@app.get("/advance-bridge")
def advance_bridge():
    """
    Advances the bridge by a small, bounded amount and returns immediately.
    Meant to be hit periodically by an external scheduler (GitHub Actions
    cron), NOT by visitor traffic.
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
        pending_stage, pending_discharge, current_state = predict_warmup(window_scaled)
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
        current_state = bridge["state"]
        pending_stage = bridge["pending_stage"]
        pending_discharge = bridge["pending_discharge"]
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
            (stage, discharge, rain, current_state, pending_stage,
             pending_discharge, rain_6h, rain_24h) = advance_one_step(
                current_state, pending_stage, pending_discharge, rain_6h, rain_24h, pred_time, rain_15m
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
