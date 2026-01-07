import numpy as np
from sklearn.preprocessing import normalize as sk_normalize

def apply(X, mode='l2', **kwargs):
    """
    Normalization.
    mode: 'l2', 'l1', 'max', 'zscore' (zscore handles separately or via SNV usually, but implemented here if requested)
    """
    if mode == 'zscore':
        # Same as SNV essentially
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1e-10
        return (X - mean) / std
    
    return sk_normalize(X, norm=mode, axis=1)

def transform(X_train, X_test, **kwargs):
    return apply(X_train, **kwargs), apply(X_test, **kwargs)
