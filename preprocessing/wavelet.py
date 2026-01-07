import numpy as np
import pywt
from typing import Optional

def apply(X: np.ndarray, wavelet: str = 'db4', level: int = 1, mode: str = 'soft', **kwargs) -> np.ndarray:
    """
    Apply Wavelet Denoising (VisuShrink / BayesShrink style simple thresholding).
    
    Args:
        X: Spectra matrix (n_samples, n_features)
        wavelet: Wavelet name (e.g., 'db4', 'sym8', 'haar')
        level: Decomposition level
        mode: Thresholding mode ('soft' or 'hard')
    """
    X_denoised = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        # Decompose
        coeffs = pywt.wavedec(X[i, :], wavelet, level=level)
        
        # Calculate threshold (VisuShrink universal threshold)
        # Sigma estimation using MAD of detailed coefficients at level 1
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(X.shape[1]))
        
        # Thresholding
        new_coeffs = []
        new_coeffs.append(coeffs[0]) # Keep approximation coefficients
        for detail in coeffs[1:]:
            new_coeffs.append(pywt.threshold(detail, threshold, mode=mode))
            
        # Reconstruct
        X_denoised[i, :] = pywt.waverec(new_coeffs, wavelet)[:X.shape[1]]
        
    return X_denoised

def transform(X_train, X_test, **kwargs):
    # Independent application
    return apply(X_train, **kwargs), apply(X_test, **kwargs)
