import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import chi2
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs" / "plots" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mahalanobis_distance(scores):
    """Compute Mahalanobis distance for PCA scores."""
    mean = np.mean(scores, axis=0)
    cov = np.cov(scores, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    diff = scores - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))


def pca_diagnostics(X: np.ndarray, dataset_name: str, save_plots: bool = True):
    """
    Execute PCA diagnostics and save:
    - Scores PC1-PC2
    - Hotelling's T²
    - Q-residuals
    - Mahalanobis Distance
    """

    print(f"\n🔍 PCA Diagnostics — {dataset_name}")

    # Fit PCA
    pca = PCA(n_components=5)
    scores = pca.fit_transform(X)
    X_recon = pca.inverse_transform(scores)

    # Q-residuals
    q_residuals = np.sum((X - X_recon) ** 2, axis=1)

    # Hotelling’s T²
    t_scores = scores / np.std(scores, axis=0)
    t2 = np.sum(t_scores ** 2, axis=1)

    # Mahalanobis
    mahal = mahalanobis_distance(scores)

    # Outlier thresholds
    alpha = 0.95
    t2_threshold = chi2.ppf(alpha, df=2)
    q_threshold = np.percentile(q_residuals, 95)
    mahal_threshold = np.percentile(mahal, 95)

    outliers = np.where(
        (t2 > t2_threshold) | (q_residuals > q_threshold) | (mahal > mahal_threshold)
    )[0]

    print(f"⚠ Outliers detectados: {outliers.tolist() if len(outliers) else 'Nenhum'}")

    # ------- PLOTS -------
    if save_plots:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f"PCA Diagnostics — {dataset_name}")

        # Scores plot
        var_exp = pca.explained_variance_ratio_ * 100
        axs[0, 0].scatter(scores[:, 0], scores[:, 1], c='teal', alpha=0.6, s=15, edgecolors='k', linewidth=0.3)
        axs[0, 0].set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
        axs[0, 0].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
        axs[0, 0].set_title("Scores Plot (PC1 vs PC2)")
        axs[0, 0].grid(True, linestyle=':', alpha=0.5)

        # Hotelling T²
        axs[0, 1].scatter(range(len(t2)), t2, s=10)
        axs[0, 1].axhline(t2_threshold, color="r", linestyle="--")
        axs[0, 1].set_title("Hotelling T²")

        # Q Residuals
        axs[1, 0].scatter(range(len(q_residuals)), q_residuals, s=10)
        axs[1, 0].axhline(q_threshold, color="r", linestyle="--")
        axs[1, 0].set_title("Q Residuals")

        # Mahalanobis
        axs[1, 1].scatter(range(len(mahal)), mahal, s=10)
        axs[1, 1].axhline(mahal_threshold, color="r", linestyle="--")
        axs[1, 1].set_title("Mahalanobis Distance")

        fig.tight_layout()
        plt.savefig(OUT_DIR / f"{dataset_name}_pca_diagnostics.png", dpi=300)
        plt.close()

    return {
        "scores": scores,
        "t2": t2,
        "q": q_residuals,
        "mahal": mahal,
        "outliers": outliers.tolist()
    }


if __name__ == "__main__":
    print("This file is meant to be imported inside the main pipeline.")
