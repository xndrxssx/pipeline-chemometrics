import numpy as np
from scipy.signal import savgol_filter

def apply(X, window_length=11, polyorder=2, deriv=0, **kwargs):
    """
    Applies Savitzky-Golay filter.
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)

def transform(X_train, X_test, **kwargs):
    return apply(X_train, **kwargs), apply(X_test, **kwargs)
