# Software and Data Availability


- **Name of software:** LGSCA-InSAR
- **Developer:** Menghui Zhi
- **Contact:** zhi_mh@163.com
- **Date first available:** September 27, 2026
- **Software required:** Python 3.12.4; TensorFlow 2.21.0; Keras 3.15.1; NumPy; pandas; scikit-learn
- **Programming language:** Python
- **Source code:** https://github.com/menghui-zhi/LGSCA-InSAR
- **Documentation:** Installation, input-data schema, execution order, and reproducibility limitations are documented in the repository README and `docs/REPRODUCIBILITY.md`.
- **Example data:** A synthetic, non-confidential trajectory dataset is provided for software installation and schema testing. It is not used to reproduce the manuscript forecasting-accuracy or timing results.
- **Public source data:** Sentinel-1 observations are publicly available through Copernicus data services.
- **Restricted research data:** The processed InSAR deformation products and GNSS observations used in the study contain project-restricted information and are therefore not publicly released. Exact GNSS monitoring-site coordinates are withheld in accordance with project confidentiality requirements.
- **Reproducibility scope:** The public repository provides the routing, model-training, rolling-origin evaluation, robustness, bootstrap, inference-timing, and global-MLP baseline workflows. Exact numerical reproduction of the manuscript results requires the restricted processed trajectory dataset and the corresponding manuscript execution conditions. Model-only inference timing is hardware- and software-environment-dependent.
