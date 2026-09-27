#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight installation and input-schema smoke test.

This does not reproduce manuscript accuracy values. It checks parsing, routing,
window construction, and manuscript model instantiation.
"""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
BASE = HERE / "SCI_Final_Robustness_v5.py"
spec = importlib.util.spec_from_file_location("robust_v5_public", BASE)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import {BASE}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=HERE.parent/"data"/"example_trajectories_small.csv")
    a=p.parse_args()
    cols=base.discover_time_columns(str(a.data))
    if len(cols)!=base.N_TIME_EXPECTED:
        raise ValueError(f"Expected 84 numeric epoch columns, found {len(cols)}")
    df=pd.read_csv(a.data,usecols=cols,dtype={c:np.float32 for c in cols})
    raw=df[cols].to_numpy(dtype=np.float32)
    base.impute_training_period_inplace(raw,base.TRAIN_END)
    r2=base.row_linear_r2_chunked(raw[:,:base.TRAIN_END],chunk_size=10000)
    starts=base.training_starts()
    X,y=base.build_windows(raw[:,:base.TRAIN_END],starts)
    g=base.build_same_dcnn_lstm(); m=base.build_raw_bp()
    print("Smoke test passed.")
    print(f"Input trajectories: {len(raw):,}")
    print(f"Epoch columns: {len(cols)}")
    print(f"High-linearity (R^2 >= 0.85): {int((r2>=base.ROUTING_R2_THRESHOLD).sum()):,}")
    print(f"Historical windows: X={X.shape}, y={y.shape}")
    print(f"Global CNN-BiLSTM parameters: {g.count_params():,}")
    print(f"Lightweight MLP parameters: {m.count_params():,}")
if __name__=="__main__": main()
