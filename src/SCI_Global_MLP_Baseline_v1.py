# -*- coding: utf-8 -*-
r"""
SCI Global MLP Baseline v1
==========================

Purpose
-------
Add ONE frozen baseline experiment to the already completed
SCI_Final_Robustness_v5 protocol:

    Global MLP baseline:
        train the SAME 3,363-parameter MLP on ALL 30,000 trajectories
        in each pre-specified robustness run, then evaluate it globally.

This script does NOT:
- change the R2 router or threshold;
- tune any hyperparameter;
- retrain the existing Global CNN-BiLSTM;
- alter the existing Selective strategy;
- search for a new model.

It reuses the exact helper functions, data path, seeds, sampling rule,
scaler, training windows, rolling-origin test design, and MLP
hyperparameters defined in SCI_Final_Robustness_v5.py.

Scientific question
-------------------
Can the lightweight MLP simply replace the CNN-BiLSTM everywhere?

If Global MLP is clearly worse overall (especially on the general subset),
while the selective strategy remains competitive, this supports the need
for selective rather than global simplification.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 0. Locate and import the frozen v5 script from the SAME folder.
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

BASE_SCRIPT = HERE / "SCI_Final_Robustness_v5.py"
if not BASE_SCRIPT.exists():
    candidates = [
        p for p in HERE.glob("SCI_Final_Robustness_v5*.py")
        if p.name != Path(__file__).name
    ]
    if len(candidates) == 1:
        BASE_SCRIPT = candidates[0]
    elif len(candidates) > 1:
        # Prefer the exact canonical name if available; otherwise newest.
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        BASE_SCRIPT = candidates[0]
    else:
        raise FileNotFoundError(
            "SCI_Final_Robustness_v5.py was not found in the same folder.\n"
            "Place this baseline script beside the frozen v5 script and rerun."
        )

spec = importlib.util.spec_from_file_location("robust_v5", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import frozen v5 script: {BASE_SCRIPT}")

base = importlib.util.module_from_spec(spec)
sys.modules["robust_v5"] = base
spec.loader.exec_module(base)

# ---------------------------------------------------------------------
# 1. Output folder. Original v5 outputs are NEVER overwritten.
# ---------------------------------------------------------------------
PUBLIC_ROOT = HERE.parent
OUTPUT_DIR = PUBLIC_ROOT / "outputs" / "global_mlp_baseline_v1"
RUN_DIR = OUTPUT_DIR / "runs"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------
def force_gc():
    gc.collect()


def original_run_folder(run_cfg: dict) -> Path:
    run_name = str(run_cfg["Run"])
    model_seed = int(run_cfg["ModelSeed"])
    sample_seed = int(run_cfg["SampleSeed"])
    return base.RUN_DIR / f"Run_{run_name}_model{model_seed}_sample{sample_seed}"


def load_original_v5_rows(run_cfg: dict) -> Dict[str, pd.Series]:
    """Load frozen Global and Selective metrics from the completed v5 run."""
    folder = original_run_folder(run_cfg)
    main_path = folder / "run_metrics.csv"
    steady_path = folder / "steady_metrics.csv"
    meta_path = folder / "run_metadata.json"

    missing = [p for p in (main_path, steady_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The frozen v5 run outputs are required before this baseline is run.\n"
            f"Missing for Run {run_cfg['Run']}:\n" +
            "\n".join(str(p) for p in missing)
        )

    main_df = pd.read_csv(main_path)
    steady_df = pd.read_csv(steady_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    global_row = main_df[
        main_df["Strategy"] == "Global SAME D-CNN-LSTM"
    ].iloc[0]

    selective_row = main_df[
        main_df["Strategy"] == "Selective Raw-BP Override"
    ].iloc[0]

    steady_global = steady_df[
        steady_df["Model"] == "Global SAME D-CNN-LSTM"
    ].iloc[0]

    steady_bp = steady_df[
        steady_df["Model"] == "Raw BP"
    ].iloc[0]

    return {
        "global": global_row,
        "selective": selective_row,
        "steady_global": steady_global,
        "steady_bp": steady_bp,
        "meta": meta,
    }


def derive_general_mae_rmse(
    overall_row: pd.Series,
    steady_row: pd.Series,
    n_total: int,
    n_steady: int,
) -> Dict[str, float]:
    """
    Derive exact general-subset MAE/MSE/RMSE from overall + steady metrics.

    This is valid here because every sampled trajectory contributes the same
    number of rolling-origin forecast scalars (12 origins x 3 horizons).
    R2 is intentionally NOT derived because pooled R2 is not linearly
    decomposable.
    """
    n_general = n_total - n_steady
    if n_general <= 0:
        raise ValueError("General subset is empty.")

    overall_mae = float(overall_row["MAE_mm"])
    steady_mae = float(steady_row["MAE_mm"])

    if "MSE_mm2" in overall_row.index:
        overall_mse = float(overall_row["MSE_mm2"])
    else:
        overall_mse = float(overall_row["RMSE_mm"]) ** 2

    if "MSE_mm2" in steady_row.index:
        steady_mse = float(steady_row["MSE_mm2"])
    else:
        steady_mse = float(steady_row["RMSE_mm"]) ** 2

    general_mae = (
        overall_mae * n_total - steady_mae * n_steady
    ) / n_general

    general_mse = (
        overall_mse * n_total - steady_mse * n_steady
    ) / n_general

    general_mse = max(general_mse, 0.0)

    return {
        "R2": np.nan,
        "MAE_mm": float(general_mae),
        "RMSE_mm": float(np.sqrt(general_mse)),
        "MSE_mm2": float(general_mse),
    }


def mean_std_table(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base_row = {c: k for c, k in zip(group_cols, keys)}

        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            row = dict(base_row)
            row.update({
                "Metric": metric,
                "Mean": float(np.mean(vals)),
                "Std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "Min": float(np.min(vals)),
                "Max": float(np.max(vals)),
                "N_Runs": int(len(vals)),
            })
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. One Global-MLP baseline run
# ---------------------------------------------------------------------
def run_one_global_mlp(
    run_cfg: dict,
    raw: np.ndarray,
    steady_all_idx: np.ndarray,
    nonlin_all_idx: np.ndarray,
    scaler,
    full_steady_count: int,
    full_nonlin_count: int,
):
    run_name = str(run_cfg["Run"])
    model_seed = int(run_cfg["ModelSeed"])
    sample_seed = int(run_cfg["SampleSeed"])

    run_folder = RUN_DIR / f"Run_{run_name}_model{model_seed}_sample{sample_seed}"
    run_folder.mkdir(parents=True, exist_ok=True)

    ck_overall = run_folder / "global_mlp_overall.csv"
    ck_regime = run_folder / "global_mlp_regime.csv"
    ck_compare = run_folder / "three_strategy_overall.csv"
    ck_delta = run_folder / "global_mlp_deltas.csv"
    ck_meta = run_folder / "metadata.json"

    required = [ck_overall, ck_regime, ck_compare, ck_delta, ck_meta]
    if all(p.exists() for p in required):
        print(f"[Resume] Run {run_name} baseline already complete.")
        return {
            "overall": pd.read_csv(ck_overall),
            "regime": pd.read_csv(ck_regime),
            "three": pd.read_csv(ck_compare),
            "delta": pd.read_csv(ck_delta),
            "meta": json.loads(ck_meta.read_text(encoding="utf-8")),
        }

    print("\n" + "=" * 112)
    print(
        f"GLOBAL MLP BASELINE — RUN {run_name}: "
        f"model_seed={model_seed}, sample_seed={sample_seed}"
    )
    print("=" * 112)

    # Exact same stratified sample rule as frozen v5.
    rng = np.random.default_rng(sample_seed)

    sample_steady_n = int(round(
        base.MODEL_POINT_LIMIT * full_steady_count /
        (full_steady_count + full_nonlin_count)
    ))
    sample_steady_n = max(1, sample_steady_n)
    sample_nonlin_n = base.MODEL_POINT_LIMIT - sample_steady_n

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

    future_fills_s = base.causal_fill_future_sample_inplace(
        steady_raw, base.TRAIN_END
    )
    future_fills_n = base.causal_fill_future_sample_inplace(
        nonlin_raw, base.TRAIN_END
    )

    steady_scaled = scaler.transform(steady_raw)
    nonlin_scaled = scaler.transform(nonlin_raw)
    all_scaled = np.concatenate([steady_scaled, nonlin_scaled], axis=0)

    tr_starts = base.training_starts()
    te_starts = base.test_starts()
    n_origins = len(te_starts)

    # Train SAME MLP architecture/hyperparameters, but now on ALL 30k trajectories.
    Xtr_all, ytr_all = base.build_windows(
        all_scaled[:, :base.TRAIN_END],
        tr_starts,
    )

    print(
        f"Training Global MLP on ALL {base.MODEL_POINT_LIMIT:,} trajectories "
        f"({len(Xtr_all):,} historical windows) ..."
    )

    base.tf.keras.backend.clear_session()
    global_mlp, fit_time, epochs_used = base.train_bp_model(
        Xtr_all,
        ytr_all,
        model_seed,
    )

    model_path = MODEL_DIR / f"run_{run_name}_global_mlp_seed{model_seed}.keras"
    global_mlp.save(model_path)

    del Xtr_all, ytr_all
    force_gc()

    # Exact same rolling-origin test inputs.
    Xte_s, yte_s = base.build_windows(steady_scaled, te_starts)
    Xte_n, yte_n = base.build_windows(nonlin_scaled, te_starts)

    yte_s_mm = scaler.inverse(yte_s)
    yte_n_mm = scaler.inverse(yte_n)

    pred_mlp_s, inf_s = base.timed_predict(
        global_mlp,
        Xte_s,
        batch_size=1024,
    )
    pred_mlp_n, inf_n = base.timed_predict(
        global_mlp,
        Xte_n,
        batch_size=1024,
    )

    pred_mlp_s_mm = scaler.inverse(pred_mlp_s)
    pred_mlp_n_mm = scaler.inverse(pred_mlp_n)

    y_all_mm = np.concatenate([
        yte_s_mm.reshape(-1),
        yte_n_mm.reshape(-1),
    ])
    pred_mlp_all_mm = np.concatenate([
        pred_mlp_s_mm.reshape(-1),
        pred_mlp_n_mm.reshape(-1),
    ])

    m_mlp_all = base.metrics(y_all_mm, pred_mlp_all_mm)
    m_mlp_s = base.metrics(yte_s_mm, pred_mlp_s_mm)
    m_mlp_n = base.metrics(yte_n_mm, pred_mlp_n_mm)

    # Load frozen v5 Global + Selective results for the same run.
    old = load_original_v5_rows(run_cfg)

    # Safety: verify exact sample counts are unchanged.
    old_meta = old["meta"]
    if int(old_meta["SampleTotal"]) != base.MODEL_POINT_LIMIT:
        raise RuntimeError(
            f"Run {run_name}: frozen v5 SampleTotal differs from current protocol."
        )
    if int(old_meta["SampleSteady"]) != sample_steady_n:
        raise RuntimeError(
            f"Run {run_name}: steady sample count differs from frozen v5."
        )
    if int(old_meta["SampleNonlinear"]) != sample_nonlin_n:
        raise RuntimeError(
            f"Run {run_name}: general sample count differs from frozen v5."
        )

    # Original global general-subset MAE/RMSE can be derived exactly.
    g_general = derive_general_mae_rmse(
        old["global"],
        old["steady_global"],
        base.MODEL_POINT_LIMIT,
        sample_steady_n,
    )

    global_mlp_overall = pd.DataFrame([{
        "Run": run_name,
        "ModelSeed": model_seed,
        "SampleSeed": sample_seed,
        "Strategy": "Global MLP",
        **m_mlp_all,
    }])

    regime_df = pd.DataFrame([
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Regime": "High-linearity",
            "Model": "Global CNN-BiLSTM",
            "N_Trajectories": sample_steady_n,
            "R2": float(old["steady_global"]["R2"]),
            "MAE_mm": float(old["steady_global"]["MAE_mm"]),
            "RMSE_mm": float(old["steady_global"]["RMSE_mm"]),
            "MSE_mm2": float(old["steady_global"].get(
                "MSE_mm2", float(old["steady_global"]["RMSE_mm"]) ** 2
            )),
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Regime": "High-linearity",
            "Model": "Selective MLP (trained on high-linearity only)",
            "N_Trajectories": sample_steady_n,
            "R2": float(old["steady_bp"]["R2"]),
            "MAE_mm": float(old["steady_bp"]["MAE_mm"]),
            "RMSE_mm": float(old["steady_bp"]["RMSE_mm"]),
            "MSE_mm2": float(old["steady_bp"].get(
                "MSE_mm2", float(old["steady_bp"]["RMSE_mm"]) ** 2
            )),
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Regime": "High-linearity",
            "Model": "Global MLP",
            "N_Trajectories": sample_steady_n,
            **m_mlp_s,
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Regime": "General",
            "Model": "Global CNN-BiLSTM",
            "N_Trajectories": sample_nonlin_n,
            **g_general,
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Regime": "General",
            "Model": "Global MLP",
            "N_Trajectories": sample_nonlin_n,
            **m_mlp_n,
        },
    ])

    three_strategy_df = pd.DataFrame([
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Strategy": "Global CNN-BiLSTM",
            "R2": float(old["global"]["R2"]),
            "MAE_mm": float(old["global"]["MAE_mm"]),
            "RMSE_mm": float(old["global"]["RMSE_mm"]),
            "MSE_mm2": float(old["global"].get(
                "MSE_mm2", float(old["global"]["RMSE_mm"]) ** 2
            )),
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Strategy": "Selective",
            "R2": float(old["selective"]["R2"]),
            "MAE_mm": float(old["selective"]["MAE_mm"]),
            "RMSE_mm": float(old["selective"]["RMSE_mm"]),
            "MSE_mm2": float(old["selective"].get(
                "MSE_mm2", float(old["selective"]["RMSE_mm"]) ** 2
            )),
        },
        {
            "Run": run_name,
            "ModelSeed": model_seed,
            "SampleSeed": sample_seed,
            "Strategy": "Global MLP",
            **m_mlp_all,
        },
    ])

    delta_df = pd.DataFrame([{
        "Run": run_name,
        "ModelSeed": model_seed,
        "SampleSeed": sample_seed,
        "Definition": "Global MLP - Global CNN-BiLSTM",
        "DeltaR2": float(m_mlp_all["R2"] - float(old["global"]["R2"])),
        "DeltaMAE_mm": float(m_mlp_all["MAE_mm"] - float(old["global"]["MAE_mm"])),
        "DeltaRMSE_mm": float(m_mlp_all["RMSE_mm"] - float(old["global"]["RMSE_mm"])),
        "General_DeltaMAE_mm": float(
            m_mlp_n["MAE_mm"] - g_general["MAE_mm"]
        ),
        "General_DeltaRMSE_mm": float(
            m_mlp_n["RMSE_mm"] - g_general["RMSE_mm"]
        ),
        "HighLinearity_GlobalMLP_DeltaMAE_vs_GlobalCNN_mm": float(
            m_mlp_s["MAE_mm"] - float(old["steady_global"]["MAE_mm"])
        ),
        "HighLinearity_GlobalMLP_DeltaRMSE_vs_GlobalCNN_mm": float(
            m_mlp_s["RMSE_mm"] - float(old["steady_global"]["RMSE_mm"])
        ),
    }])

    meta = {
        "Run": run_name,
        "ModelSeed": model_seed,
        "SampleSeed": sample_seed,
        "SampleTotal": base.MODEL_POINT_LIMIT,
        "SampleHighLinearity": sample_steady_n,
        "SampleGeneral": sample_nonlin_n,
        "TrainingWindows": int(base.MODEL_POINT_LIMIT * len(tr_starts)),
        "TestOrigins": int(n_origins),
        "GlobalMLPParameters": int(global_mlp.count_params()),
        "GlobalMLPFitTime_s": float(fit_time),
        "GlobalMLPEpochs": int(epochs_used),
        "GlobalMLPInference_s": float(inf_s + inf_n),
        "FutureCausalFillsHighLinearity": int(future_fills_s),
        "FutureCausalFillsGeneral": int(future_fills_n),
        "ModelPath": str(model_path),
        "Note": (
            "Same frozen MLP architecture/hyperparameters as selective MLP; "
            "only training scope changed from high-linearity subset to all 30k trajectories."
        ),
    }

    global_mlp_overall.to_csv(ck_overall, index=False, encoding="utf-8-sig")
    regime_df.to_csv(ck_regime, index=False, encoding="utf-8-sig")
    three_strategy_df.to_csv(ck_compare, index=False, encoding="utf-8-sig")
    delta_df.to_csv(ck_delta, index=False, encoding="utf-8-sig")
    ck_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nThree-strategy overall comparison:")
    print(three_strategy_df.to_string(index=False))
    print("\nGlobal MLP - Global CNN-BiLSTM deltas:")
    print(delta_df.to_string(index=False))

    del global_mlp
    del steady_raw, nonlin_raw, steady_scaled, nonlin_scaled, all_scaled
    del Xte_s, Xte_n, yte_s, yte_n
    del pred_mlp_s, pred_mlp_n
    force_gc()
    base.tf.keras.backend.clear_session()

    return {
        "overall": global_mlp_overall,
        "regime": regime_df,
        "three": three_strategy_df,
        "delta": delta_df,
        "meta": meta,
    }


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Public-release command line
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen global-MLP baseline after the robustness protocol."
    )
    parser.add_argument("--data", type=Path, default=Path(base.DATA_PATH))
    parser.add_argument(
        "--robustness-output", type=Path, default=base.OUTPUT_DIR,
        help="Completed output directory from SCI_Final_Robustness_v5.py."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def configure_public_paths(args):
    global OUTPUT_DIR, RUN_DIR, MODEL_DIR
    base.DATA_PATH = str(args.data.expanduser().resolve())
    base.OUTPUT_DIR = args.robustness_output.expanduser().resolve()
    base.RUN_DIR = base.OUTPUT_DIR / "runs"
    base.MODEL_DIR = base.OUTPUT_DIR / "models"

    OUTPUT_DIR = args.output.expanduser().resolve()
    RUN_DIR = OUTPUT_DIR / "runs"
    MODEL_DIR = OUTPUT_DIR / "models"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    configure_public_paths(args)

    print("=" * 116)
    print("SCI GLOBAL MLP BASELINE v1")
    print("=" * 116)
    print(f"Frozen v5 script : {BASE_SCRIPT}")
    print(f"Data             : {base.DATA_PATH}")
    print(f"Original v5 dir  : {base.OUTPUT_DIR}")
    print(f"Baseline output  : {OUTPUT_DIR}")
    print("Question         : Can the same lightweight MLP replace the global CNN-BiLSTM everywhere?")
    print("=" * 116)

    if not Path(base.DATA_PATH).exists():
        raise FileNotFoundError(base.DATA_PATH)

    # Load full domain exactly as frozen v5.
    print("\n[1/4] Loading full domain and reproducing frozen routing ...")

    time_cols = base.discover_time_columns(base.DATA_PATH)
    df = pd.read_csv(
        base.DATA_PATH,
        usecols=time_cols,
        dtype={c: np.float32 for c in time_cols},
    )
    df = df[time_cols]
    raw = df.values.astype(np.float32, copy=False)
    del df
    force_gc()

    n_points, n_time = raw.shape
    if n_time != base.N_TIME_EXPECTED:
        raise ValueError(
            f"Expected {base.N_TIME_EXPECTED} epochs, found {n_time}."
        )

    n_train_nan_rows = base.impute_training_period_inplace(
        raw, base.TRAIN_END
    )

    scaler = base.ScalarMinMax(
        float(np.nanmin(raw[:, :base.TRAIN_END])),
        float(np.nanmax(raw[:, :base.TRAIN_END])),
    )

    route_r2 = base.row_linear_r2_chunked(raw[:, :base.TRAIN_END])
    steady_mask = route_r2 >= base.ROUTING_R2_THRESHOLD
    nonlin_mask = ~steady_mask

    full_steady_count = int(steady_mask.sum())
    full_nonlin_count = int(nonlin_mask.sum())

    steady_all_idx = np.flatnonzero(steady_mask)
    nonlin_all_idx = np.flatnonzero(nonlin_mask)

    print(
        f"Frozen routing reproduced: high-linearity={full_steady_count:,} "
        f"({full_steady_count/n_points*100:.6f}%), "
        f"general={full_nonlin_count:,} "
        f"({full_nonlin_count/n_points*100:.6f}%)"
    )

    del route_r2, steady_mask, nonlin_mask
    force_gc()

    # Run A/B/C using the same frozen seeds.
    print("\n[2/4] Training only the added Global MLP baseline ...")
    results = []
    for cfg in base.RUNS:
        results.append(
            run_one_global_mlp(
                cfg,
                raw,
                steady_all_idx,
                nonlin_all_idx,
                scaler,
                full_steady_count,
                full_nonlin_count,
            )
        )

    # Aggregate.
    print("\n[3/4] Aggregating baseline results ...")

    all_three = pd.concat([r["three"] for r in results], ignore_index=True)
    all_regime = pd.concat([r["regime"] for r in results], ignore_index=True)
    all_delta = pd.concat([r["delta"] for r in results], ignore_index=True)
    all_meta = pd.DataFrame([r["meta"] for r in results])

    mean_std_overall = mean_std_table(
        all_three,
        ["Strategy"],
        ["R2", "MAE_mm", "RMSE_mm"],
    )

    mean_std_regime = mean_std_table(
        all_regime,
        ["Regime", "Model"],
        ["R2", "MAE_mm", "RMSE_mm"],
    )

    delta_summary = mean_std_table(
        all_delta,
        ["Definition"],
        [
            "DeltaR2",
            "DeltaMAE_mm",
            "DeltaRMSE_mm",
            "General_DeltaMAE_mm",
            "General_DeltaRMSE_mm",
            "HighLinearity_GlobalMLP_DeltaMAE_vs_GlobalCNN_mm",
            "HighLinearity_GlobalMLP_DeltaRMSE_vs_GlobalCNN_mm",
        ],
    )

    # Direction counts.
    mlp_worse_mae = int((all_delta["DeltaMAE_mm"] > 0).sum())
    mlp_worse_rmse = int((all_delta["DeltaRMSE_mm"] > 0).sum())
    mlp_worse_general_mae = int((all_delta["General_DeltaMAE_mm"] > 0).sum())
    mlp_worse_general_rmse = int((all_delta["General_DeltaRMSE_mm"] > 0).sum())

    # Save.
    print("\n[4/4] Saving baseline package ...")

    all_three.to_csv(
        OUTPUT_DIR / "01_per_run_three_strategy_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_regime.to_csv(
        OUTPUT_DIR / "02_per_run_regime_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_delta.to_csv(
        OUTPUT_DIR / "03_global_mlp_vs_global_cnn_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mean_std_overall.to_csv(
        OUTPUT_DIR / "04_three_strategy_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mean_std_regime.to_csv(
        OUTPUT_DIR / "05_regime_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_summary.to_csv(
        OUTPUT_DIR / "06_delta_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_meta.to_csv(
        OUTPUT_DIR / "07_training_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = f"""
SCI GLOBAL MLP BASELINE v1
==========================

FROZEN DESIGN
Full domain: {n_points:,} trajectories x {n_time} epochs
Routing threshold: R2 >= {base.ROUTING_R2_THRESHOLD:.2f}
High-linearity: {full_steady_count:,} ({full_steady_count/n_points*100:.6f}%)
General: {full_nonlin_count:,} ({full_nonlin_count/n_points*100:.6f}%)
Each run: {base.MODEL_POINT_LIMIT:,} stratified trajectories
Forecast: {base.N_INPUT}->{base.N_OUTPUT}, rolling-origin epochs 71-84
Added baseline: SAME 3,363-parameter MLP trained on ALL 30k trajectories
No threshold/model/hyperparameter tuning was performed.

PER-RUN THREE-STRATEGY OVERALL METRICS
{all_three.to_string(index=False)}

GLOBAL MLP - GLOBAL CNN-BiLSTM DELTAS
Positive MAE/RMSE delta means Global MLP is worse.
{all_delta.to_string(index=False)}

THREE-STRATEGY MEAN ± STD
{mean_std_overall.to_string(index=False)}

REGIME MEAN ± STD
{mean_std_regime.to_string(index=False)}

DIRECTION COUNTS
Global MLP MAE worse than Global CNN-BiLSTM: {mlp_worse_mae}/3 runs
Global MLP RMSE worse than Global CNN-BiLSTM: {mlp_worse_rmse}/3 runs
General-subset Global MLP MAE worse: {mlp_worse_general_mae}/3 runs
General-subset Global MLP RMSE worse: {mlp_worse_general_rmse}/3 runs

INTERPRETATION
- Strong support for selective simplification:
  Global MLP is worse overall/general in most or all runs, while the existing
  selective strategy remains close to the Global CNN-BiLSTM.
- Mixed support:
  Global MLP is only slightly worse or directions vary across runs.
- Challenge to the routing rationale:
  Global MLP matches or beats Global CNN-BiLSTM globally across runs.

IMPORTANT
This is a one-baseline falsification check, NOT a new model search.
Do not tune the MLP or the routing threshold based on these test results.
"""

    (OUTPUT_DIR / "08_summary.txt").write_text(
        summary.strip() + "\n",
        encoding="utf-8",
    )

    metadata = {
        "version": "SCI_Global_MLP_Baseline_v1",
        "base_script": str(BASE_SCRIPT),
        "data_path": str(base.DATA_PATH),
        "full_domain_points": int(n_points),
        "time_steps": int(n_time),
        "training_nan_rows_imputed": int(n_train_nan_rows),
        "router_threshold": float(base.ROUTING_R2_THRESHOLD),
        "high_linearity_count": int(full_steady_count),
        "general_count": int(full_nonlin_count),
        "runs": base.RUNS,
        "question": (
            "Can the frozen 3,363-parameter MLP replace the global CNN-BiLSTM everywhere?"
        ),
        "frozen_rule": (
            "Same MLP architecture/hyperparameters; only training scope changes "
            "from high-linearity subset to all 30k trajectories."
        ),
    }

    (OUTPUT_DIR / "09_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 116)
    print(summary)
    print("=" * 116)
    print("\nPlease upload these files after completion:")
    for name in [
        "01_per_run_three_strategy_overall.csv",
        "02_per_run_regime_metrics.csv",
        "03_global_mlp_vs_global_cnn_deltas.csv",
        "04_three_strategy_mean_std.csv",
        "05_regime_mean_std.csv",
        "06_delta_mean_std.csv",
        "07_training_metadata.csv",
        "08_summary.txt",
        "09_metadata.json",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
