# Reproducibility scope

## What this public release provides
The repository provides the core software workflow used in the manuscript:

- future-blind trajectory routing based on straight-line R² from epochs 1–70;
- the fixed R² threshold of 0.85;
- the global attention-enhanced CNN–BiLSTM forecaster;
- the lightweight MLP selective override;
- 10-to-3 historical training-window construction;
- rolling-origin evaluation over epochs 71–84;
- three pre-specified robustness runs with frozen sampling/model seeds;
- R², MAE, and RMSE evaluation in physical units;
- point-cluster paired bootstrap analysis;
- model-only inference timing; and
- the global-MLP baseline used to test uniform lightweight replacement.

The public-release code changes are limited to removal of local computer paths and addition of command-line path configuration. The scientific protocol is intentionally frozen.

## What is not publicly distributed
The repository does not distribute the processed 1,979,024-trajectory InSAR deformation dataset or the GNSS observations used in the manuscript. These products contain project-restricted information. Exact GNSS monitoring-site coordinates are also withheld under project confidentiality requirements.

The small CSV under `data/` is synthetic and non-confidential. It is provided only to verify installation, input parsing, historical R² routing, window generation, and model construction.

## Numerical reproduction
Exact manuscript numerical results cannot be reproduced from the public synthetic example alone. Numerical reproduction requires access to the restricted processed trajectory dataset and the manuscript execution conditions.

Run A of the original robustness workflow was allowed to reuse validated seed-42 v4 checkpoints when available. In the public release, if those optional checkpoints are not supplied, Run A is trained fresh with the same frozen architecture and seed. This preserves the scientific protocol but may prevent bit-for-bit identity with an execution that reused the original checkpoint.

Deep-learning results can also vary slightly across TensorFlow builds, numerical libraries, operating systems, and hardware even when random seeds are fixed.

## Timing reproduction
The paper reports **model-only inference timing**, not end-to-end system timing. Those timing measurements are hardware-dependent.

The manuscript timing environment was:

- Windows 11;
- Intel Core Ultra 5 225H CPU (14 cores);
- 31.5 GB system memory;
- Python 3.12.4;
- TensorFlow 2.21.0;
- Keras 3.15.1; and
- no TensorFlow-detected GPU device.

Timing measurements from a different CPU, GPU, TensorFlow build, operating system, thread configuration, or background workload should not be expected to match the paper exactly.

## Terminology
Some original implementation variables and output labels use `steady` and `nonlinear`. To minimize changes to validated code, these names are retained. Their manuscript equivalents are:

- `steady` → **high-linearity trajectories**;
- `nonlinear` → **general trajectories**; and
- `Raw BP` → **lightweight MLP**.
