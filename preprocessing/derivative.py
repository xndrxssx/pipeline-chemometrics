import numpy as np
from scipy.signal import savgol_filter

def transform(X_train, X_test, order=1, window_length=5, polyorder=2, method='diff'):
    """
    Calculates the derivative of spectral data.
    
    Parameters:
    - order: Derivative order (1 or 2)
    - method: 'diff' (simple difference) or 'sg' (Savitzky-Golay)
    - window_length: Window size for SG (only used if method='sg')
    - polyorder: Polynomial order for SG (only used if method='sg')
    """
    X_train_d = X_train.copy()
    X_test_d = X_test.copy()
    
    if method == 'sg':
        # Use Savitzky-Golay for derivative
        X_train_d = savgol_filter(X_train, window_length=window_length, polyorder=polyorder, deriv=order, axis=1)
        X_test_d = savgol_filter(X_test, window_length=window_length, polyorder=polyorder, deriv=order, axis=1)
    else:
        # Use NumPy Gradient (Central Difference)
        # Note: np.gradient returns a list if axis is not specified, or array if axis is.
        # For order=1
        X_train_d = np.gradient(X_train, axis=1)
        X_test_d = np.gradient(X_test, axis=1)
        
        if order == 2:
            X_train_d = np.gradient(X_train_d, axis=1)
            X_test_d = np.gradient(X_test_d, axis=1)

    return X_train_d, X_test_d
