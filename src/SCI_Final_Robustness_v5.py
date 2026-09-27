# -*- coding: utf-8 -*-
r"""
SCI Final Robustness v5
=======================
Final robustness experiment before freezing the manuscript.

Purpose
-------
Test whether the v4.1 conclusion remains directionally stable under
independent repetitions with different sampling / model random seeds.

NO NEW MODEL SEARCH.
NO VMD.
NO THRESHOLD TUNING.
NO PCA ROUTING.

Fixed scientific protocol
-------------------------
- Full domain: current 1,979,024 x 84 InSAR trajectories
- Router: per-trajectory straight-line R2 from epochs 1-70 only
- Fixed threshold: R2 >= 0.85 -> steady
- Forecast task: 10 -> 3
- Training windows: epochs 1-70 only
- Test: rolling-origin epochs 71-84
- Global model: SAME D-CNN-LSTM used in v4/v4.1
- Selective model:
      steady    -> Raw BP
      nonlinear -> Global D-CNN-LSTM
- Common scalar MinMax scaler fitted from full-domain epochs 1-70 only
- All primary metrics are computed in physical mm.

Terminology note for the public release
---------------------------------------
- Historical implementation name "steady" = manuscript "high-linearity" subset.
- Historical implementation name "nonlinear" = manuscript "general" subset.

Pre-specified repetitions
-------------------------
Run A:
    model_seed = 42
    sample_seed = 542
    Reuses validated v4/v4.1 models if available.

Run B:
    model_seed = 2026
    sample_seed = 2026

Run C:
    model_seed = 3407
    sample_seed = 3407

Each run uses a fresh stratified 30,000-point sample.
Runs B/C train fresh Global D-CNN-LSTM and Raw BP.

Main question
-------------
Across the three independent runs, does Selective Raw-BP Override consistently:
- maintain / improve R2?
- reduce MAE?
- reduce RMSE?
- improve steady-regime prediction?
- reduce test inference cost?

Resume behavior
---------------
Each completed run writes its own checkpoint CSV.
If the program is interrupted after a run finishes, rerunning the script
will reuse the completed run rather than retraining it.

"""

from __future__ import annotations

# =============================================================================
# 0. PYCHARM / KERAS PATH SANITIZER
# =============================================================================
import os
import sys

_BAD = ("keras\\src\\backend", "keras/src/backend")
sys.path[:] = [
    p for p in sys.path
    if not any(bad in str(p).lower() for bad in _BAD)
]

if "PYTHONPATH" in os.environ:
    parts = os.environ["PYTHONPATH"].split(os.pathsep)
    parts = [
        p for p in parts
        if not any(bad in str(p).lower() for bad in _BAD)
    ]
    if parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        os.environ.pop("PYTHONPATH", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import gc
import re
import json
import time
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Input, Reshape, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
        Flatten, Activation, RepeatVector, Permute, Multiply, Bidirectional
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except Exception as e:
    raise ImportError(f"[v5] TensorFlow import failed: {e}")

warnings.filterwarnings("ignore")

# =============================================================================
# 2. CONFIGURATION -- DO NOT TUNE ON TEST RESULTS
# =============================================================================
PUBLIC_ROOT = Path(__file__).resolve().parents[1]

# Public-release defaults. Override them with command-line arguments if needed.
DATA_PATH = str(PUBLIC_ROOT / "data" / "input_trajectories.csv")
V4_DIR = PUBLIC_ROOT / "optional_pretrained" / "v4"
OUTPUT_DIR = PUBLIC_ROOT / "outputs" / "robustness_v5"
RUN_DIR = OUTPUT_DIR / "runs"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = 70
N_TIME_EXPECTED = 84
ROUTING_R2_THRESHOLD = 0.85

N_INPUT = 10
N_OUTPUT = 3
TRAIN_STRIDE = 2

MODEL_POINT_LIMIT = 30000

# Three pre-specified repetitions.
RUNS = [
    {
        "Run": "A",
        "ModelSeed": 42,
        "SampleSeed": 542,
        "ReuseV4": True,
    },
    {
        "Run": "B",
        "ModelSeed": 2026,
        "SampleSeed": 2026,
        "ReuseV4": False,
    },
    {
        "Run": "C",
        "ModelSeed": 3407,
        "SampleSeed": 3407,
        "ReuseV4": False,
    },
]

# D-CNN-LSTM: identical to v4/v4.1.
DCNN_EPOCHS = 35
DCNN_BATCH = 1024
DCNN_PATIENCE = 6
DCNN_VALIDATION_SPLIT = 0.15
DCNN_LR = 0.0003

# Raw BP: identical to v4/v4.1.
BP_EPOCHS = 40
BP_BATCH = 256
BP_PATIENCE = 8
BP_VALIDATION_SPLIT = 0.15
BP_LR = 0.0005

# v4 saved seed-42 models.
V4_GLOBAL_MODEL = V4_DIR / "models" / "global_same_dcnn.keras"
V4_RAW_BP_MODEL = V4_DIR / "models" / "steady_raw_bp.keras"

# Optional point-cluster bootstrap within each run.
# This is descriptive robustness support, not threshold/model selection.
BOOTSTRAP_REPS = 2000

# =============================================================================
# 3. HELPERS
# =============================================================================
def force_gc():
    gc.collect()


class ScalarMinMax:
    def __init__(self, data_min: float, data_max: float):
        self.data_min = float(data_min)
        self.data_max = float(data_max)

    def transform(self, x: np.ndarray) -> np.ndarray:
        den = self.data_max - self.data_min
        if abs(den) < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return (
            2.0 * (x.astype(np.float32) - self.data_min) / den - 1.0
        ).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        den = self.data_max - self.data_min
        return (
            (z.astype(np.float64) + 1.0) * 0.5 * den + self.data_min
        )


def extract_time_step(col_name: str) -> int:
    nums = re.findall(r"\d+", str(col_name))
    return int(nums[-1]) if nums else 0


def discover_time_columns(csv_path: str) -> List[str]:
    sample = pd.read_csv(csv_path, nrows=10)
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    for c in ("x", "y"):
        if c in numeric_cols:
            numeric_cols.remove(c)
    return sorted(numeric_cols, key=extract_time_step)


def impute_training_period_inplace(raw: np.ndarray, train_end: int) -> int:
    train_view = raw[:, :train_end]
    bad = np.flatnonzero(np.isnan(train_view).any(axis=1))
    if len(bad) == 0:
        return 0

    print(
        f"[Imputation] {len(bad):,} rows contain NaNs in epochs 1-{train_end}; "
        "interpolating within training period only."
    )

    temp = pd.DataFrame(train_view[bad])
    temp.interpolate(
        method="linear", axis=1, limit_direction="both", inplace=True
    )
    temp.ffill(axis=1, inplace=True)
    temp.bfill(axis=1, inplace=True)

    raw[bad, :train_end] = temp.values.astype(np.float32)

    del temp
    force_gc()
    return len(bad)


def causal_fill_future_sample_inplace(sample: np.ndarray, train_end: int) -> int:
    fills = 0
    for j in range(train_end, sample.shape[1]):
        mask = np.isnan(sample[:, j])
        if np.any(mask):
            fills += int(mask.sum())
            sample[mask, j] = sample[mask, j - 1]

    if np.isnan(sample).any():
        raise RuntimeError(
            "NaNs remain after training-only interpolation and causal future fill."
        )

    return fills


def row_linear_r2_chunked(
    matrix: np.ndarray,
    chunk_size: int = 100000
) -> np.ndarray:
    n, T = matrix.shape

    x = np.arange(T, dtype=np.float64)
    xc = x - x.mean()
    sxx = np.sum(xc * xc)

    out = np.empty(n, dtype=np.float32)
    t0 = time.perf_counter()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        y = matrix[start:end].astype(np.float64, copy=False)
        ym = np.mean(y, axis=1, keepdims=True)
        yc = y - ym

        num = np.sum(yc * xc[None, :], axis=1)
        syy = np.sum(yc * yc, axis=1)
        den = sxx * syy

        r2 = np.zeros(end - start, dtype=np.float64)
        good = den > 1e-20
        r2[good] = (num[good] ** 2) / den[good]

        out[start:end] = np.clip(r2, 0.0, 1.0).astype(np.float32)

        if start == 0 or end == n or ((start // chunk_size) % 5 == 0):
            print(
                f"    routing R2 {end:,}/{n:,} ({end/n:.1%}), "
                f"elapsed={time.perf_counter()-t0:.1f}s"
            )

    return out


def training_starts() -> List[int]:
    max_start = TRAIN_END - N_INPUT - N_OUTPUT  # 57
    starts = list(range(0, max_start + 1, TRAIN_STRIDE))
    if max_start not in starts:
        starts.append(max_start)
    return sorted(set(starts))


def test_starts() -> List[int]:
    first = TRAIN_END - N_INPUT  # 60
    last = N_TIME_EXPECTED - N_INPUT - N_OUTPUT  # 71
    return list(range(first, last + 1))


def build_windows(
    series_scaled: np.ndarray,
    starts: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.concatenate(
        [series_scaled[:, s:s + N_INPUT] for s in starts],
        axis=0
    ).astype(np.float32, copy=False)

    y = np.concatenate(
        [
            series_scaled[:, s + N_INPUT:s + N_INPUT + N_OUTPUT]
            for s in starts
        ],
        axis=0
    ).astype(np.float32, copy=False)

    return X, y


def metrics(
    y_true_mm: np.ndarray,
    y_pred_mm: np.ndarray,
) -> Dict[str, float]:
    yt = np.asarray(y_true_mm).reshape(-1)
    yp = np.asarray(y_pred_mm).reshape(-1)

    mse = mean_squared_error(yt, yp)

    return {
        "R2": float(r2_score(yt, yp)),
        "MAE_mm": float(mean_absolute_error(yt, yp)),
        "RMSE_mm": float(np.sqrt(mse)),
        "MSE_mm2": float(mse),
    }


def point_level_metrics(
    y_true_mm: np.ndarray,
    y_pred_mm: np.ndarray,
    n_points: int,
    n_origins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    yt = y_true_mm.reshape(
        n_origins, n_points, N_OUTPUT
    ).transpose(1, 0, 2)

    yp = y_pred_mm.reshape(
        n_origins, n_points, N_OUTPUT
    ).transpose(1, 0, 2)

    err = yp - yt
    mae = np.mean(np.abs(err), axis=(1, 2))
    rmse = np.sqrt(np.mean(err ** 2, axis=(1, 2)))

    return mae, rmse


def paired_bootstrap_mean_difference(
    baseline: np.ndarray,
    candidate: np.ndarray,
    reps: int,
    seed: int,
) -> Dict[str, float]:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)

    if baseline.shape != candidate.shape:
        raise ValueError("Bootstrap arrays have different shapes.")

    diff = candidate - baseline
    n = len(diff)

    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=np.float64)

    chunk_reps = 100
    done = 0

    while done < reps:
        take = min(chunk_reps, reps - done)
        idx = rng.integers(
            0, n, size=(take, n), endpoint=False
        )
        boot[done:done+take] = np.mean(diff[idx], axis=1)
        done += take

    lo, hi = np.quantile(boot, [0.025, 0.975])

    return {
        "MeanDifference": float(np.mean(diff)),
        "CI95_Lower": float(lo),
        "CI95_Upper": float(hi),
        "N_Points": int(n),
        "BootstrapReps": int(reps),
    }


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# =============================================================================
# 4. MODEL DEFINITIONS
# =============================================================================
def build_same_dcnn_lstm() -> Model:
    inputs = Input(shape=(N_INPUT,), name="Input")
    x = Reshape((N_INPUT, 1))(inputs)

    x = Conv1D(
        64,
        kernel_size=2,
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(1e-5),
    )(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(
        128,
        kernel_size=2,
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(1e-5),
    )(x)

    x = Conv1D(
        256,
        kernel_size=2,
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L2(1e-5),
    )(x)

    attention = Dense(1, activation="tanh")(x)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)
    attention = RepeatVector(x.shape[-1])(attention)
    attention = Permute([2, 1])(attention)
    x = Multiply()([x, attention])

    x = Bidirectional(
        LSTM(
            128,
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.2,
        )
    )(x)

    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(N_OUTPUT)(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Robustness-SAME-D-CNN-LSTM",
    )

    model.compile(
        optimizer=Adam(learning_rate=DCNN_LR),
        loss="mse",
    )

    return model


def build_raw_bp() -> Sequential:
    model = Sequential(name="Robustness-Steady-Raw-BP")

    model.add(Input(shape=(N_INPUT,)))
    model.add(
        Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-6),
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(N_OUTPUT))

    model.compile(
        optimizer=Adam(learning_rate=BP_LR),
        loss="mse",
    )

    return model


def train_global_model(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    seed: int,
) -> Tuple[Model, float, int]:
    tf.keras.backend.clear_session()
    set_all_seeds(seed)

    model = build_same_dcnn_lstm()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=DCNN_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    t0 = time.perf_counter()

    hist = model.fit(
        Xtr,
        ytr,
        epochs=DCNN_EPOCHS,
        batch_size=DCNN_BATCH,
        validation_split=DCNN_VALIDATION_SPLIT,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    dt = time.perf_counter() - t0
    return model, dt, len(hist.history["loss"])


def train_bp_model(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    seed: int,
) -> Tuple[Model, float, int]:
    # Do not clear the Keras session here: the freshly trained Global D-CNN
    # is still needed for prediction in this same robustness run.
    set_all_seeds(seed)

    model = build_raw_bp()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=BP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    t0 = time.perf_counter()

    hist = model.fit(
        Xtr,
        ytr,
        epochs=BP_EPOCHS,
        batch_size=BP_BATCH,
        validation_split=BP_VALIDATION_SPLIT,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    dt = time.perf_counter() - t0
    return model, dt, len(hist.history["loss"])


def timed_predict(
    model: Model,
    X: np.ndarray,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, float]:
    warm_n = min(len(X), 1024)
    if warm_n:
        _ = model.predict(
            X[:warm_n],
            batch_size=batch_size,
            verbose=0,
        )

    t0 = time.perf_counter()

    pred = model.predict(
        X,
        batch_size=batch_size,
        verbose=0,
    )

    dt = time.perf_counter() - t0
    return pred.astype(np.float32), dt


# =============================================================================
# 5. ONE ROBUSTNESS RUN
# =============================================================================
def run_one(
    run_cfg: dict,
    raw: np.ndarray,
    steady_all_idx: np.ndarray,
    nonlin_all_idx: np.ndarray,
    scaler: ScalarMinMax,
    full_steady_count: int,
    full_nonlin_count: int,
    route_time: float,
):
    run_name = str(run_cfg["Run"])
    model_seed = int(run_cfg["ModelSeed"])
    sample_seed = int(run_cfg["SampleSeed"])
    reuse_v4 = bool(run_cfg["ReuseV4"])

    run_folder = RUN_DIR / f"Run_{run_name}_model{model_seed}_sample{sample_seed}"
    run_folder.mkdir(parents=True, exist_ok=True)

    checkpoint = run_folder / "run_metrics.csv"
    checkpoint_steady = run_folder / "steady_metrics.csv"
    checkpoint_horizon = run_folder / "horizon_metrics.csv"
    checkpoint_bootstrap = run_folder / "bootstrap_ci.csv"
    checkpoint_meta = run_folder / "run_metadata.json"

    # Resume: only reuse a run if all required checkpoints exist.
    required = [
        checkpoint,
        checkpoint_steady,
        checkpoint_horizon,
        checkpoint_bootstrap,
        checkpoint_meta,
    ]

    if all(p.exists() for p in required):
        print(
            f"\n[Resume] Run {run_name} already complete. "
            "Loading checkpoint instead of retraining."
        )

        return {
            "main": pd.read_csv(checkpoint),
            "steady": pd.read_csv(checkpoint_steady),
            "horizon": pd.read_csv(checkpoint_horizon),
            "bootstrap": pd.read_csv(checkpoint_bootstrap),
            "metadata": json.loads(
                checkpoint_meta.read_text(encoding="utf-8")
            ),
        }

    print("\n" + "=" * 112)
    print(
        f"RUN {run_name}: model_seed={model_seed}, "
        f"sample_seed={sample_seed}, reuse_v4={reuse_v4}"
    )
    print("=" * 112)

    # -------------------------------------------------------------------------
    # Sample stratified to full-domain regime ratio.
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(sample_seed)

    sample_steady_n = int(round(
        MODEL_POINT_LIMIT * full_steady_count /
        (full_steady_count + full_nonlin_count)
    ))
    sample_steady_n = max(1, sample_steady_n)
    sample_nonlin_n = MODEL_POINT_LIMIT - sample_steady_n

    steady_idx = np.sort(
        rng.choice(
            steady_all_idx,
            size=sample_steady_n,
            replace=False,
        )
    ).astype(np.int64)

    nonlin_idx = np.sort(
        rng.choice(
            nonlin_all_idx,
            size=sample_nonlin_n,
            replace=False,
        )
    ).astype(np.int64)

    steady_raw = raw[steady_idx].copy()
    nonlin_raw = raw[nonlin_idx].copy()

    future_fills_s = causal_fill_future_sample_inplace(
        steady_raw, TRAIN_END
    )
    future_fills_n = causal_fill_future_sample_inplace(
        nonlin_raw, TRAIN_END
    )

    steady_scaled = scaler.transform(steady_raw)
    nonlin_scaled = scaler.transform(nonlin_raw)
    all_scaled = np.concatenate(
        [steady_scaled, nonlin_scaled],
        axis=0,
    )

    tr_starts = training_starts()
    te_starts = test_starts()
    n_origins = len(te_starts)

    Xte_s, yte_s = build_windows(
        steady_scaled,
        te_starts,
    )

    Xte_n, yte_n = build_windows(
        nonlin_scaled,
        te_starts,
    )

    yte_s_mm = scaler.inverse(yte_s)
    yte_n_mm = scaler.inverse(yte_n)

    # -------------------------------------------------------------------------
    # Models.
    # -------------------------------------------------------------------------
    global_train_time = np.nan
    bp_train_time = np.nan
    global_epochs = np.nan
    bp_epochs = np.nan
    source = "fresh training"

    can_reuse_v4 = (
        reuse_v4
        and V4_GLOBAL_MODEL.exists()
        and V4_RAW_BP_MODEL.exists()
        and model_seed == 42
        and sample_seed == 542
    )

    if can_reuse_v4:
        source = "reused v4/v4.1 validated seed-42 models"

        print("Loading validated v4 seed-42 models ...")
        global_model = tf.keras.models.load_model(
            V4_GLOBAL_MODEL,
            compile=False,
        )
        bp_model = tf.keras.models.load_model(
            V4_RAW_BP_MODEL,
            compile=False,
        )

    else:
        # ---------------- Global D-CNN ----------------
        print("\nTraining fresh Global D-CNN-LSTM ...")

        Xtr_g, ytr_g = build_windows(
            all_scaled[:, :TRAIN_END],
            tr_starts,
        )

        global_model, global_train_time, global_epochs = train_global_model(
            Xtr_g,
            ytr_g,
            model_seed,
        )

        del Xtr_g, ytr_g
        force_gc()

        global_model.save(
            MODEL_DIR /
            f"run_{run_name}_global_seed{model_seed}.keras"
        )

        # ---------------- Steady BP ----------------
        print("\nTraining fresh steady Raw-BP ...")

        Xtr_s, ytr_s = build_windows(
            steady_scaled[:, :TRAIN_END],
            tr_starts,
        )

        bp_model, bp_train_time, bp_epochs = train_bp_model(
            Xtr_s,
            ytr_s,
            model_seed,
        )

        del Xtr_s, ytr_s
        force_gc()

        bp_model.save(
            MODEL_DIR /
            f"run_{run_name}_rawbp_seed{model_seed}.keras"
        )

    # -------------------------------------------------------------------------
    # Predictions.
    # -------------------------------------------------------------------------
    pred_g_s, inf_g_s = timed_predict(
        global_model,
        Xte_s,
        batch_size=2048,
    )

    pred_g_n, inf_g_n = timed_predict(
        global_model,
        Xte_n,
        batch_size=2048,
    )

    pred_bp_s, inf_bp_s = timed_predict(
        bp_model,
        Xte_s,
        batch_size=1024,
    )

    pred_g_s_mm = scaler.inverse(pred_g_s)
    pred_g_n_mm = scaler.inverse(pred_g_n)
    pred_bp_s_mm = scaler.inverse(pred_bp_s)

    # -------------------------------------------------------------------------
    # Main physical-mm metrics.
    # -------------------------------------------------------------------------
    y_all_mm = np.concatenate([
        yte_s_mm.reshape(-1),
        yte_n_mm.reshape(-1),
    ])

    pred_global_all = np.concatenate([
        pred_g_s_mm.reshape(-1),
        pred_g_n_mm.reshape(-1),
    ])

    pred_selective_all = np.concatenate([
        pred_bp_s_mm.reshape(-1),
        pred_g_n_mm.reshape(-1),
    ])

    m_global = metrics(
        y_all_mm,
        pred_global_all,
    )

    m_selective = metrics(
        y_all_mm,
        pred_selective_all,
    )

    main_df = pd.DataFrame([
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Strategy": "Global SAME D-CNN-LSTM",
            **m_global,
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Strategy": "Selective Raw-BP Override",
            **m_selective,
        },
    ])

    # Exact steady set.
    steady_df = pd.DataFrame([
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Model": "Global SAME D-CNN-LSTM",
            **metrics(yte_s_mm, pred_g_s_mm),
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Model": "Raw BP",
            **metrics(yte_s_mm, pred_bp_s_mm),
        },
    ])

    # Horizon metrics.
    horizon_rows = []

    for strategy, ps, pn in [
        (
            "Global SAME D-CNN-LSTM",
            pred_g_s_mm,
            pred_g_n_mm,
        ),
        (
            "Selective Raw-BP Override",
            pred_bp_s_mm,
            pred_g_n_mm,
        ),
    ]:
        for h in range(N_OUTPUT):
            yt = np.concatenate([
                yte_s_mm[:, h],
                yte_n_mm[:, h],
            ])

            yp = np.concatenate([
                ps[:, h],
                pn[:, h],
            ])

            horizon_rows.append({
                "Run": run_name,
                "ModelSeed": model_seed,
                "SampleSeed": sample_seed,
                "Strategy": strategy,
                "Horizon": f"H{h+1}",
                **metrics(yt, yp),
            })

    horizon_df = pd.DataFrame(horizon_rows)

    # -------------------------------------------------------------------------
    # Point-cluster paired bootstrap.
    # -------------------------------------------------------------------------
    g_s_mae, g_s_rmse = point_level_metrics(
        yte_s_mm,
        pred_g_s_mm,
        sample_steady_n,
        n_origins,
    )

    g_n_mae, g_n_rmse = point_level_metrics(
        yte_n_mm,
        pred_g_n_mm,
        sample_nonlin_n,
        n_origins,
    )

    bp_s_mae, bp_s_rmse = point_level_metrics(
        yte_s_mm,
        pred_bp_s_mm,
        sample_steady_n,
        n_origins,
    )

    baseline_mae_all = np.concatenate([
        g_s_mae,
        g_n_mae,
    ])

    baseline_rmse_all = np.concatenate([
        g_s_rmse,
        g_n_rmse,
    ])

    selective_mae_all = np.concatenate([
        bp_s_mae,
        g_n_mae,
    ])

    selective_rmse_all = np.concatenate([
        bp_s_rmse,
        g_n_rmse,
    ])

    bootstrap_rows = []

    comparisons = [
        (
            "Overall 30k points",
            "PointMAE_mm",
            baseline_mae_all,
            selective_mae_all,
        ),
        (
            "Overall 30k points",
            "PointRMSE_mm",
            baseline_rmse_all,
            selective_rmse_all,
        ),
        (
            "Steady exact set",
            "PointMAE_mm",
            g_s_mae,
            bp_s_mae,
        ),
        (
            "Steady exact set",
            "PointRMSE_mm",
            g_s_rmse,
            bp_s_rmse,
        ),
    ]

    for j, (scope, metric_name, base, cand) in enumerate(comparisons):
        row = paired_bootstrap_mean_difference(
            base,
            cand,
            reps=BOOTSTRAP_REPS,
            seed=model_seed + 1000 + j,
        )

        row.update({
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Scope": scope,
            "Metric": metric_name,
            "Reference": "Global SAME D-CNN-LSTM",
            "Candidate": "Selective Raw-BP Override",
            "DifferenceDefinition": "candidate - global; negative is better",
        })

        bootstrap_rows.append(row)

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # -------------------------------------------------------------------------
    # Run-level deltas / metadata.
    # -------------------------------------------------------------------------
    global_params = int(global_model.count_params())
    bp_params = int(bp_model.count_params())

    p_steady = full_steady_count / (
        full_steady_count + full_nonlin_count
    )

    population_weighted_active_params = (
        p_steady * bp_params +
        (1.0 - p_steady) * global_params
    )

    run_meta = {
        "Run": run_name,
        "ModelSeed": model_seed,
        "SampleSeed": sample_seed,
        "ModelSource": source,
        "SampleTotal": MODEL_POINT_LIMIT,
        "SampleSteady": sample_steady_n,
        "SampleNonlinear": sample_nonlin_n,
        "FutureCausalFillsSteady": future_fills_s,
        "FutureCausalFillsNonlinear": future_fills_n,
        "GlobalParameters": global_params,
        "BPParameters": bp_params,
        "PopulationWeightedActiveParameters": float(
            population_weighted_active_params
        ),
        "ActiveParameterReduction_vs_Global_pct": float(
            (
                1.0 -
                population_weighted_active_params / global_params
            ) * 100.0
        ),
        "GlobalFitTime_s": (
            float(global_train_time)
            if np.isfinite(global_train_time)
            else None
        ),
        "BPFitTime_s": (
            float(bp_train_time)
            if np.isfinite(bp_train_time)
            else None
        ),
        "GlobalEpochs": (
            int(global_epochs)
            if np.isfinite(global_epochs)
            else None
        ),
        "BPEpochs": (
            int(bp_epochs)
            if np.isfinite(bp_epochs)
            else None
        ),
        "GlobalInference_s": float(
            inf_g_s + inf_g_n
        ),
        "SelectiveInference_s": float(
            inf_bp_s + inf_g_n
        ),
        "RouterFullDomain_s": float(route_time),
        "DeltaR2_SelectiveMinusGlobal": float(
            m_selective["R2"] - m_global["R2"]
        ),
        "DeltaMAE_SelectiveMinusGlobal_mm": float(
            m_selective["MAE_mm"] - m_global["MAE_mm"]
        ),
        "DeltaRMSE_SelectiveMinusGlobal_mm": float(
            m_selective["RMSE_mm"] - m_global["RMSE_mm"]
        ),
    }

    # Save checkpoints.
    main_df.to_csv(
        checkpoint,
        index=False,
        encoding="utf-8-sig",
    )

    steady_df.to_csv(
        checkpoint_steady,
        index=False,
        encoding="utf-8-sig",
    )

    horizon_df.to_csv(
        checkpoint_horizon,
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_df.to_csv(
        checkpoint_bootstrap,
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint_meta.write_text(
        json.dumps(
            run_meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nRUN {run_name} COMPLETE")
    print(main_df.to_string(index=False))
    print(
        f"Delta Selective-Global: "
        f"R2={run_meta['DeltaR2_SelectiveMinusGlobal']:+.6f}, "
        f"MAE={run_meta['DeltaMAE_SelectiveMinusGlobal_mm']:+.6f} mm, "
        f"RMSE={run_meta['DeltaRMSE_SelectiveMinusGlobal_mm']:+.6f} mm"
    )

    del global_model, bp_model
    del steady_raw, nonlin_raw, steady_scaled, nonlin_scaled, all_scaled
    del Xte_s, Xte_n, yte_s, yte_n
    del pred_g_s, pred_g_n, pred_bp_s
    force_gc()
    tf.keras.backend.clear_session()

    return {
        "main": main_df,
        "steady": steady_df,
        "horizon": horizon_df,
        "bootstrap": bootstrap_df,
        "metadata": run_meta,
    }


# =============================================================================
# 6. FINAL AGGREGATION
# =============================================================================
def mean_std_table(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    rows = []

    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {
            col: key
            for col, key in zip(group_cols, keys)
        }

        for metric in metric_cols:
            values = group[metric].astype(float).values

            row = dict(base)
            row["Metric"] = metric
            row["Mean"] = float(np.mean(values))
            row["Std"] = float(np.std(values, ddof=1))
            row["Min"] = float(np.min(values))
            row["Max"] = float(np.max(values))
            row["N_Runs"] = len(values)

            rows.append(row)

    return pd.DataFrame(rows)



# =============================================================================
# PUBLIC-RELEASE COMMAND LINE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen robustness protocol for the linearity-guided "
            "selective complexity-allocation framework."
        )
    )
    parser.add_argument(
        "--data", type=Path, default=Path(DATA_PATH),
        help="CSV with 84 numeric deformation-epoch columns; optional x/y are ignored."
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help="Directory for robustness outputs and trained models."
    )
    parser.add_argument(
        "--v4-dir", type=Path, default=V4_DIR,
        help=(
            "Optional validated v4 checkpoint directory. If checkpoints are absent, "
            "Run A is trained fresh with the manuscript seed."
        )
    )
    return parser.parse_args()


def configure_public_paths(args):
    global DATA_PATH, V4_DIR, OUTPUT_DIR, RUN_DIR, MODEL_DIR
    global V4_GLOBAL_MODEL, V4_RAW_BP_MODEL

    DATA_PATH = str(args.data.expanduser().resolve())
    V4_DIR = args.v4_dir.expanduser().resolve()
    OUTPUT_DIR = args.output.expanduser().resolve()
    RUN_DIR = OUTPUT_DIR / "runs"
    MODEL_DIR = OUTPUT_DIR / "models"
    V4_GLOBAL_MODEL = V4_DIR / "models" / "global_same_dcnn.keras"
    V4_RAW_BP_MODEL = V4_DIR / "models" / "steady_raw_bp.keras"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    configure_public_paths(args)

    print("=" * 116)
    print("SCI FINAL ROBUSTNESS v5")
    print("=" * 116)
    print(f"Python     : {sys.executable}")
    print(f"TensorFlow : {tf.__version__}")
    print(f"Data       : {DATA_PATH}")
    print(f"Output     : {OUTPUT_DIR}")
    print("Runs       : A(42/542), B(2026/2026), C(3407/3407)")
    print("=" * 116)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    # -------------------------------------------------------------------------
    # Load full domain once.
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading full domain and freezing router ...")

    time_cols = discover_time_columns(DATA_PATH)

    df = pd.read_csv(
        DATA_PATH,
        usecols=time_cols,
        dtype={c: np.float32 for c in time_cols},
    )

    df = df[time_cols]
    raw = df.values.astype(np.float32, copy=False)

    del df
    force_gc()

    n_points, n_time = raw.shape

    if n_time != N_TIME_EXPECTED:
        raise ValueError(
            f"Expected {N_TIME_EXPECTED} time steps, found {n_time}."
        )

    n_train_nan_rows = impute_training_period_inplace(
        raw,
        TRAIN_END,
    )

    scaler = ScalarMinMax(
        float(np.nanmin(raw[:, :TRAIN_END])),
        float(np.nanmax(raw[:, :TRAIN_END])),
    )

    route_t0 = time.perf_counter()

    route_r2 = row_linear_r2_chunked(
        raw[:, :TRAIN_END]
    )

    route_time = time.perf_counter() - route_t0

    steady_mask = route_r2 >= ROUTING_R2_THRESHOLD
    nonlinear_mask = ~steady_mask

    full_steady_count = int(steady_mask.sum())
    full_nonlin_count = int(nonlinear_mask.sum())

    steady_all_idx = np.flatnonzero(steady_mask)
    nonlin_all_idx = np.flatnonzero(nonlinear_mask)

    print(
        f"\nFrozen routing: steady={full_steady_count:,} "
        f"({full_steady_count/n_points*100:.6f}%), "
        f"nonlinear={full_nonlin_count:,} "
        f"({full_nonlin_count/n_points*100:.6f}%)"
    )

    del route_r2, steady_mask, nonlinear_mask
    force_gc()

    # -------------------------------------------------------------------------
    # Execute / resume three runs.
    # -------------------------------------------------------------------------
    print("\n[2/4] Running three pre-specified repetitions ...")

    results = []

    for cfg in RUNS:
        results.append(
            run_one(
                cfg,
                raw,
                steady_all_idx,
                nonlin_all_idx,
                scaler,
                full_steady_count,
                full_nonlin_count,
                route_time,
            )
        )

    # -------------------------------------------------------------------------
    # Aggregate.
    # -------------------------------------------------------------------------
    print("\n[3/4] Aggregating robustness results ...")

    main_all = pd.concat(
        [r["main"] for r in results],
        ignore_index=True,
    )

    steady_all = pd.concat(
        [r["steady"] for r in results],
        ignore_index=True,
    )

    horizon_all = pd.concat(
        [r["horizon"] for r in results],
        ignore_index=True,
    )

    bootstrap_all = pd.concat(
        [r["bootstrap"] for r in results],
        ignore_index=True,
    )

    metadata_all = pd.DataFrame(
        [r["metadata"] for r in results]
    )

    # Per-run paired deltas.
    delta_rows = []

    for run_name in main_all["Run"].unique():
        sub = main_all[
            main_all["Run"] == run_name
        ]

        g = sub[
            sub["Strategy"] ==
            "Global SAME D-CNN-LSTM"
        ].iloc[0]

        s = sub[
            sub["Strategy"] ==
            "Selective Raw-BP Override"
        ].iloc[0]

        # steady exact set
        ss = steady_all[
            steady_all["Run"] == run_name
        ]

        sg = ss[
            ss["Model"] ==
            "Global SAME D-CNN-LSTM"
        ].iloc[0]

        sb = ss[
            ss["Model"] == "Raw BP"
        ].iloc[0]

        delta_rows.append({
            "Run": run_name,
            "ModelSeed": int(g["ModelSeed"]),
            "SampleSeed": int(g["SampleSeed"]),

            "Overall_DeltaR2": float(
                s["R2"] - g["R2"]
            ),
            "Overall_DeltaMAE_mm": float(
                s["MAE_mm"] - g["MAE_mm"]
            ),
            "Overall_DeltaRMSE_mm": float(
                s["RMSE_mm"] - g["RMSE_mm"]
            ),

            "Steady_DeltaR2": float(
                sb["R2"] - sg["R2"]
            ),
            "Steady_DeltaMAE_mm": float(
                sb["MAE_mm"] - sg["MAE_mm"]
            ),
            "Steady_DeltaRMSE_mm": float(
                sb["RMSE_mm"] - sg["RMSE_mm"]
            ),

            "SelectiveBetter_R2": bool(
                s["R2"] >= g["R2"]
            ),
            "SelectiveBetter_MAE": bool(
                s["MAE_mm"] <= g["MAE_mm"]
            ),
            "SelectiveBetter_RMSE": bool(
                s["RMSE_mm"] <= g["RMSE_mm"]
            ),
        })

    delta_df = pd.DataFrame(delta_rows)

    # Mean ± std tables.
    overall_mean_std = mean_std_table(
        main_all,
        ["Strategy"],
        ["R2", "MAE_mm", "RMSE_mm"],
    )

    steady_mean_std = mean_std_table(
        steady_all,
        ["Model"],
        ["R2", "MAE_mm", "RMSE_mm"],
    )

    horizon_mean_std = mean_std_table(
        horizon_all,
        ["Strategy", "Horizon"],
        ["R2", "MAE_mm", "RMSE_mm"],
    )

    # Delta summary is computed directly across the three paired runs.
    delta_summary_rows = []
    for metric in [
        "Overall_DeltaR2",
        "Overall_DeltaMAE_mm",
        "Overall_DeltaRMSE_mm",
        "Steady_DeltaR2",
        "Steady_DeltaMAE_mm",
        "Steady_DeltaRMSE_mm",
    ]:
        vals = delta_df[metric].astype(float).values
        delta_summary_rows.append({
            "Metric": metric,
            "Mean": float(np.mean(vals)),
            "Std": float(np.std(vals, ddof=1)),
            "Min": float(np.min(vals)),
            "Max": float(np.max(vals)),
            "N_Runs": len(vals),
        })

    delta_mean_std = pd.DataFrame(delta_summary_rows)

    # Directional consistency.
    direction_summary = pd.DataFrame([
        {
            "Criterion": "Overall Selective R2 >= Global",
            "RunsSatisfied": int(delta_df["SelectiveBetter_R2"].sum()),
            "TotalRuns": len(delta_df),
        },
        {
            "Criterion": "Overall Selective MAE <= Global",
            "RunsSatisfied": int(delta_df["SelectiveBetter_MAE"].sum()),
            "TotalRuns": len(delta_df),
        },
        {
            "Criterion": "Overall Selective RMSE <= Global",
            "RunsSatisfied": int(delta_df["SelectiveBetter_RMSE"].sum()),
            "TotalRuns": len(delta_df),
        },
        {
            "Criterion": "Steady Raw-BP R2 >= Global D-CNN",
            "RunsSatisfied": int((delta_df["Steady_DeltaR2"] >= 0).sum()),
            "TotalRuns": len(delta_df),
        },
        {
            "Criterion": "Steady Raw-BP MAE <= Global D-CNN",
            "RunsSatisfied": int((delta_df["Steady_DeltaMAE_mm"] <= 0).sum()),
            "TotalRuns": len(delta_df),
        },
        {
            "Criterion": "Steady Raw-BP RMSE <= Global D-CNN",
            "RunsSatisfied": int((delta_df["Steady_DeltaRMSE_mm"] <= 0).sum()),
            "TotalRuns": len(delta_df),
        },
    ])

    # -------------------------------------------------------------------------
    # Save final outputs.
    # -------------------------------------------------------------------------
    print("\n[4/4] Saving final robustness package ...")

    main_all.to_csv(
        OUTPUT_DIR / "01_per_run_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_df.to_csv(
        OUTPUT_DIR / "02_per_run_paired_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_mean_std.to_csv(
        OUTPUT_DIR / "03_overall_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    steady_all.to_csv(
        OUTPUT_DIR / "04_per_run_steady_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    steady_mean_std.to_csv(
        OUTPUT_DIR / "05_steady_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_all.to_csv(
        OUTPUT_DIR / "06_per_run_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_mean_std.to_csv(
        OUTPUT_DIR / "07_horizon_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_all.to_csv(
        OUTPUT_DIR / "08_per_run_bootstrap_ci.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata_all.to_csv(
        OUTPUT_DIR / "09_run_efficiency_and_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_mean_std.to_csv(
        OUTPUT_DIR / "10_delta_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    direction_summary.to_csv(
        OUTPUT_DIR / "11_directional_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_metadata = {
        "version": "SCI_Final_Robustness_v5",
        "full_domain_points": int(n_points),
        "time_steps": int(n_time),
        "train_end": TRAIN_END,
        "router": {
            "feature": "trajectory straight-line regression R2",
            "threshold": ROUTING_R2_THRESHOLD,
            "period": "epochs 1-70 only",
            "steady_count": full_steady_count,
            "steady_percent": float(
                full_steady_count / n_points * 100.0
            ),
            "nonlinear_count": full_nonlin_count,
            "nonlinear_percent": float(
                full_nonlin_count / n_points * 100.0
            ),
        },
        "forecast": {
            "input": N_INPUT,
            "output": N_OUTPUT,
            "test": "rolling-origin epochs 71-84",
            "training_stride": TRAIN_STRIDE,
        },
        "runs": RUNS,
        "sample_points_per_run": MODEL_POINT_LIMIT,
        "bootstrap_reps_per_run": BOOTSTRAP_REPS,
        "training_nan_rows_imputed": n_train_nan_rows,
        "note": (
            "This script is a robustness repetition only. "
            "No model, threshold, architecture, or VMD search is performed."
        ),
    }

    (OUTPUT_DIR / "12_metadata.json").write_text(
        json.dumps(
            final_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Compact final summary.
    r2_direction = int(delta_df["SelectiveBetter_R2"].sum())
    mae_direction = int(delta_df["SelectiveBetter_MAE"].sum())
    rmse_direction = int(delta_df["SelectiveBetter_RMSE"].sum())

    steady_r2_direction = int((delta_df["Steady_DeltaR2"] >= 0).sum())
    steady_mae_direction = int((delta_df["Steady_DeltaMAE_mm"] <= 0).sum())
    steady_rmse_direction = int((delta_df["Steady_DeltaRMSE_mm"] <= 0).sum())

    summary = f"""
SCI FINAL ROBUSTNESS v5
=======================

FIXED DESIGN
Full domain: {n_points:,} points x {n_time} epochs
Router: first70 trajectory R2 >= {ROUTING_R2_THRESHOLD:.2f}
Steady: {full_steady_count:,} ({full_steady_count/n_points*100:.6f}%)
Nonlinear: {full_nonlin_count:,} ({full_nonlin_count/n_points*100:.6f}%)
Forecast: 10->3 rolling-origin, future test epochs 71-84
Each run: 30,000 stratified points
Runs:
  A: model_seed=42, sample_seed=542
  B: model_seed=2026, sample_seed=2026
  C: model_seed=3407, sample_seed=3407

PER-RUN OVERALL METRICS
{main_all.to_string(index=False)}

PAIRED DELTAS (Selective - Global)
{delta_df.to_string(index=False)}

OVERALL MEAN ± STD
{overall_mean_std.to_string(index=False)}

STEADY EXACT-SET MEAN ± STD
{steady_mean_std.to_string(index=False)}

DELTA MEAN ± STD
{delta_mean_std.to_string(index=False)}

DIRECTIONAL CONSISTENCY
Overall:
  R2 Selective >= Global : {r2_direction}/3 runs
  MAE Selective <= Global: {mae_direction}/3 runs
  RMSE Selective <= Global: {rmse_direction}/3 runs

Steady exact set:
  Raw-BP R2 >= Global D-CNN : {steady_r2_direction}/3 runs
  Raw-BP MAE <= Global D-CNN: {steady_mae_direction}/3 runs
  Raw-BP RMSE <= Global D-CNN: {steady_rmse_direction}/3 runs

INTERPRETATION RULE
- Strong robustness support:
    selective direction is favorable in 3/3 runs for MAE/RMSE,
    and steady Raw-BP advantage is favorable in 3/3 runs.
- Partial support:
    mean metrics favor selective routing, but one run reverses direction.
- Weak support:
    directions are inconsistent and mean differences are near zero/reversed.

IMPORTANT
This is the final model robustness experiment.
Do not start a new model/threshold search based on these test results.
"""

    (OUTPUT_DIR / "13_summary.txt").write_text(
        summary.strip() + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 116)
    print(summary)
    print("=" * 116)

    print("Please upload these final robustness outputs:")
    for name in [
        "01_per_run_overall_metrics.csv",
        "02_per_run_paired_deltas.csv",
        "03_overall_mean_std.csv",
        "04_per_run_steady_metrics.csv",
        "05_steady_mean_std.csv",
        "06_per_run_horizon_metrics.csv",
        "07_horizon_mean_std.csv",
        "08_per_run_bootstrap_ci.csv",
        "09_run_efficiency_and_metadata.csv",
        "10_delta_mean_std.csv",
        "11_directional_consistency.csv",
        "12_metadata.json",
        "13_summary.txt",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
