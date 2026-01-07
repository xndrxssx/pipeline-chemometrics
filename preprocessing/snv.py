import numpy as np

def apply(X, **kwargs):
    """
    Standard Normal Variate (SNV).
    x_snv = (x - mean) / std
    """
    # Calculate mean and std along the spectral axis (axis 1)
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    
    # Avoid division by zero
    std[std == 0] = 1e-10
    
    return (X - mean) / std

def transform(X_train, X_test, **kwargs):
    return apply(X_train, **kwargs), apply(X_test, **kwargs)
