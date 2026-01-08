# 🧪 Chemometrics Pipeline (High-Performance)

An industrial-grade, modular pipeline for spectral data analysis (Chemometrics). Designed for **high-throughput screening** of Regression and Classification models, leveraging multi-core processing to exhaustively search for the best preprocessing and model hyperparameters.

---

## 🚀 Key Features

*   **Exhaustive Grid Search:** Automatically tests thousands of combinations:
    *   **Preprocessing:** Savitzky-Golay (multi-window/poly), SNV, MSC, Detrend, Wavelet Denoising, AsLS, OPLS, OSC.
    *   **Models:** PLSR, PCR, SVM (Linear/RBF), Random Forest, XGBoost, ElasticNet.
    *   **Hyperparameters:** Automated tuning via 5-Fold Cross-Validation.
*   **Leakage-Proof Architecture:**
    *   Strict split (Kennard-Stone) applied *before* any preprocessing.
    *   Data-dependent filters (MSC, OPLS, OSC) learn from Training set and project Test set correctly.
*   **Premium Reporting:**
    *   **HTML Dashboard:** Single-file interactive report (`chemometrics_report.html`) with sortable tables, filters, and embedded plots.
    *   **Excel Logs:** Detailed spreadsheets with all metrics (R², RMSE, RPD, Bias, Slope).
    *   **Visualization:** High-quality plots for PCA Diagnostics (Influence Plots), Variable Importance (Wavelength mapping), and Prediction (Calibration/Test).
*   **Hardware Optimized:** Native support for `n_jobs=-1` to utilize 100% of available CPU cores (ideal for i9/Ryzen 9).

---

## 📂 Project Structure

```text
chemometrics_pipeline/
├── config/                 # ⚙️ Configuration Center
│   ├── models.yaml         # Model selection & Hyperparameter grids
│   ├── preprocessing.yaml  # Custom preprocessing chains (Presets)
│   ├── filters.yaml        # Ranges for automated filter grid (e.g., SG windows)
│   └── data_config.py      # Dataset paths and metadata
├── datasets/               # 📁 Input Data (Excel files)
├── diagnostics/            # 🔍 Statistical tests (PCA, Outliers)
├── loaders/                # 📥 Data ingestion & formatting
├── modeling/               # 🤖 Model training wrappers
├── outputs/                # 📤 Results
│   ├── models/             # Best models saved as .joblib
│   ├── plots/              # PNG visuals (PCA, Boxplots, Scatter)
│   └── reports/            # HTML Dashboard & Excel Logs
├── plots/                  # 📊 Plotting logic (Matplotlib/Seaborn)
├── preprocessing/          # 🧹 Filter implementations
├── utils/                  # 🛠 Helpers (HTML Generator)
└── pipeline_runner.py      # ▶️ Main execution script
```

---

## ⚙️ Configuration Guide

### 1. Controlling the Search Space (`config/filters.yaml`)
Define the ranges for the **Automated Preprocessing Grid**. The pipeline will generate the Cartesian product of these options.

```yaml
sg:
  enabled: true
  windows: [11, 15, 21, 31]  # Will test all these window sizes
  polyorders: [2, 3]         # combined with these polynomial orders
  derivatives: [1, 2]        # and these derivatives

wavelet:
  enabled: true
  wavelets: ['db4', 'sym8']
  levels: [1, 2]
```

### 2. Custom Chains (`config/preprocessing.yaml`)
Define specific, complex combinations manually (Presets). These run *in addition* to the automated grid.

```yaml
presets:
  OPLS_SNV_SG_Optimized:
    - name: opls
      params: {max_components: 2}
    - name: snv
    - name: sg
      params: {window_length: 15, polyorder: 2, deriv: 1}
```

### 3. Models & Hardware (`config/models.yaml`)
Enable/Disable models and set CPU usage.

```yaml
global:
  n_jobs: -1  # USE ALL CORES (Recommended for large grids)
  cv:
    kfold: 5

models:
  xgboost:
    enabled: true
    grid:
      max_depth: [3, 5, 7]
      learning_rate: [0.01, 0.1]
```

---

## ▶️ How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Pipeline:**
    ```bash
    python pipeline_runner.py
    ```

    *The progress bar `0/97` indicates the number of **Preprocessing Chains** being tested. Inside each chain, multiple models and hyperparameters are evaluated.*

3.  **Analyze Results:**
    *   Open `outputs/reports/chemometrics_report.html` in your browser.
    *   Use the **Filters** to drill down by Dataset or Model.
    *   Click **"Details"** on any row to see the prediction plot and full configuration.

---

## 📊 Outputs Explained

### PCA Diagnostics
*   **Scores Plot:** Visualizes sample distribution. Colors represent **Mahalanobis Distance** (yellower = more extreme).
*   **Influence Plot:** **Hotelling T²** (X-axis) vs **Q-Residuals** (Y-axis). Helps identify outliers:
    *   High T²: Leverage points (extreme but valid).
    *   High Q: Spectral outliers (invalid/noisy).

### Prediction Plots
*   **Blue:** Calibration samples (Training).
*   **Red:** Validation samples (CV).
*   **Green:** External Test samples (Never seen during training).
*   **Fit Line (Red):** The model's trend line.

### Outlier Detection
*   **Horizontal Boxplots:** Show distribution of target variables (Train vs Test). Black dots represent individual samples.

---

## 🛠 Troubleshooting

*   **"Method not found":** Ensure the method name in YAML matches the Python module name in `preprocessing/`.
*   **Memory Issues:** If `n_jobs=-1` crashes your system (rare with 128GB RAM but possible), reduce it to `n_jobs=10` in `config/models.yaml`.
*   **Slow Execution:** Disable computationally expensive filters (like Wavelet or large OPLS grids) in `filters.yaml` or reduce the number of Model Hyperparameters.

---

**Author:** Luyza
**Date:** January 2026