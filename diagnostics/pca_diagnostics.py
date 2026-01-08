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
        
        # 1. Scores Plot (PC1 vs PC2)
        plt.figure(figsize=(10, 6))
        var_exp = pca.explained_variance_ratio_ * 100
        sc = plt.scatter(scores[:, 0], scores[:, 1], c=mahal, cmap='viridis', 
                         alpha=0.8, s=40, edgecolors='k', linewidth=0.3)
        plt.colorbar(sc, label='Mahalanobis Dist')
        
        for idx in outliers:
            plt.text(scores[idx, 0], scores[idx, 1], str(idx), fontsize=9, color='red', weight='bold')
            
        plt.xlabel(f"PC1 ({var_exp[0]:.1f}%)")
        plt.ylabel(f"PC2 ({var_exp[1]:.1f}%)")
        plt.title(f"Scores Plot: {dataset_name}")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{dataset_name}_pca_scores.png", dpi=300)
        plt.close()

        # 2. Influence Plot (T^2 vs Q-Residuals)
        plt.figure(figsize=(8, 6))
        
        # Scatter plot
        plt.scatter(t2, q_residuals, c='blue', alpha=0.6, edgecolors='k', s=30, label='Samples')
        
        # Limits
        plt.axvline(t2_threshold, color='r', linestyle='--', label=f'T² Limit ({alpha*100:.0f}%)')
        plt.axhline(q_threshold, color='r', linestyle='--', label=f'Q Limit ({alpha*100:.0f}%)')
        
        # Labels for Outliers (high T2 or high Q)
        for idx in outliers:
             if t2[idx] > t2_threshold or q_residuals[idx] > q_threshold:
                plt.text(t2[idx], q_residuals[idx], str(idx), fontsize=8, color='black', weight='bold')

        plt.xlabel('Hotelling T² (Model Distance)')
        plt.ylabel('Q-Residuals (Spectral Residual)')
        plt.title(f"Influence Plot: {dataset_name}")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{dataset_name}_pca_influence.png", dpi=300)
        plt.close()

        # 3. Mahalanobis (Optional standalone)
        plt.figure(figsize=(10, 5))
        plt.plot(mahal, 'd-', color='purple', markersize=5)
        plt.axhline(mahal_threshold, color='r', linestyle='--')
        plt.title(f"Mahalanobis Distance: {dataset_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{dataset_name}_pca_mahalanobis.png", dpi=300)
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
