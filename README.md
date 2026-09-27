# LGSCA-InSAR

Public research code accompanying **“Linearity-Guided Selective Complexity Allocation for Large-Scale InSAR-Based Land Subsidence Forecasting.”**

## Scope
This release contains the core analysis software used for the manuscript: historical-only R² routing, the global attention-enhanced CNN–BiLSTM, the lightweight MLP override, rolling-origin evaluation, three independent robustness runs, point-cluster bootstrap analysis, model-only inference timing, and the uniform global-MLP baseline.

The original implementation variable names `steady` and `nonlinear` are retained in parts of the frozen code to minimize changes to the validated experiment. In the manuscript, these correspond to **high-linearity** and **general** trajectories, respectively.

The public-release edits are engineering-only: local file paths were removed and command-line arguments were added. The manuscript scientific protocol, model definitions, thresholds, random seeds, training windows, evaluation design, and metrics were not changed.

## Repository files
- `src/SCI_Final_Robustness_v5.py`: main frozen robustness/selective-routing workflow.
- `src/SCI_Global_MLP_Baseline_v1.py`: frozen global lightweight-MLP baseline.
- `src/smoke_test.py`: quick installation, input-schema, routing, window, and model-construction check.
- `data/example_trajectories_small.csv`: synthetic non-confidential data for the smoke test only.
- `docs/REPRODUCIBILITY.md`: exact scope and limitations of numerical reproducibility.
- `docs/ENVSOFT_Software_and_Data_Availability_Draft.md`: manuscript statement template to finalize after the GitHub repository is public.

Publication-figure scripts are intentionally not included because the release is focused on the core modelling and evaluation workflow rather than figure styling.

## Installation
The manuscript timing environment used Python 3.12.4, TensorFlow 2.21.0, and Keras 3.15.1.

```bash
pip install -r requirements.txt
```

## Input format
The main workflow expects a CSV with **84 numeric deformation-epoch columns**. Optional numeric columns named `x` and `y` are treated as coordinates and excluded automatically. Epoch columns should contain sortable numeric suffixes, for example `epoch_01` ... `epoch_84`.

The public repository does **not** contain the full processed InSAR trajectory dataset used in the manuscript.

## Quick smoke test

```bash
python src/smoke_test.py
```

This checks parsing, historical-only R² routing, training-window generation, and construction of the manuscript CNN–BiLSTM and lightweight MLP. It does not train the full manuscript experiment and does not reproduce the manuscript accuracy values.

## Main robustness workflow
Use the restricted/full trajectory CSV only if you are authorized to access it.

```bash
python src/SCI_Final_Robustness_v5.py --data /path/to/input_trajectories.csv --output outputs/robustness_v5
```

The scientific settings remain the manuscript settings: 30,000 stratified trajectories per run; runs A/B/C with the frozen model/sample seeds; routing and training from epochs 1–70; 10-to-3 forecasting; and rolling-origin evaluation over epochs 71–84.

Run A can optionally reuse validated seed-42 v4 checkpoints if they are supplied under `optional_pretrained/v4/models/`. If those checkpoints are absent, the public code trains Run A fresh with the same frozen seed and architecture.

## Global MLP baseline
Run this only after the robustness workflow has completed for the same authorized input dataset:

```bash
python src/SCI_Global_MLP_Baseline_v1.py --data /path/to/input_trajectories.csv --robustness-output outputs/robustness_v5 --output outputs/global_mlp_baseline_v1
```

## Data availability and reproducibility scope
Sentinel-1 source observations are publicly available through Copernicus data services. The processed InSAR deformation products and GNSS observations used in the manuscript contain project-restricted information and are not distributed in this repository. Exact GNSS monitoring-site coordinates are also withheld because of project confidentiality requirements.

Accordingly, this repository openly provides the **analysis software and model workflow**, but it does not claim that the public synthetic example alone can reproduce the numerical results in the paper. Exact numerical reproduction requires the restricted processed trajectory dataset and the corresponding manuscript execution conditions. In addition, inference timing is hardware- and software-environment-dependent.

The manuscript timing measurements were obtained under CPU-only execution on Windows 11 with an Intel Core Ultra 5 225H processor, 31.5 GB system memory, Python 3.12.4, TensorFlow 2.21.0, and Keras 3.15.1. Timing values obtained on other systems should not be expected to match exactly.

The included synthetic dataset is intended only for software inspection and installation/schema testing. It is not a substitute for the restricted research dataset and does not reproduce the reported forecasting accuracy or inference-time results.

See `docs/REPRODUCIBILITY.md` for additional details.

## Citation
Citation metadata are provided in `CITATION.cff`. If you use this software, please cite the associated manuscript.

## License
MIT License.
