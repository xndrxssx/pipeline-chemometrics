import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs" / "plots" / "targets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def robust_zscore(series: pd.Series):
    """Robust Z Score using median and MAD."""
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return np.zeros(len(series))
    return 0.6745 * (series - median) / mad


def detect_outliers_iqr(series: pd.Series):
    """Outliers using classical IQR method."""
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return np.where((series < lower) | (series > upper))[0]


def analyze_targets(dataset_name: str, y_df: pd.DataFrame, save_plot: bool = True):
    """
    Perform:
    - Boxplot
    - IQR outlier detection
    - Robust Z-score detection
    """

    print(f"\n📊 Checking target distributions — {dataset_name}")
    results = {}

    # ---- Plot ----
    if save_plot:
        plt.figure(figsize=(12, 6))
        y_df.boxplot()
        plt.xticks(rotation=45)
        plt.title(f"Target Distribution — {dataset_name}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{dataset_name}_targets_boxplot.png", dpi=300)
        plt.close()

    # ---- Per attribute detection ----
    for col in y_df.columns:
        series = y_df[col].dropna()

        iqr_out = detect_outliers_iqr(series)
        rz = robust_zscore(series)
        rz_out = np.where(np.abs(rz) > 3)[0]

        # unify indexes
        out_union = sorted(list(set(iqr_out.tolist() + rz_out.tolist())))

        results[col] = {
            "iqr_outliers": iqr_out.tolist(),
            "robust_z_outliers": rz_out.tolist(),
            "combined": out_union
        }

        if len(out_union):
            print(f"⚠ Outliers in {col}: {out_union}")
        else:
            print(f"✔ {col}: no flagged outliers.")

    return results


if __name__ == "__main__":
    print("This module should be called from pipeline or notebook.")
