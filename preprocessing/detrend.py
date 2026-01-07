import numpy as np
from scipy.signal import detrend as scipy_detrend

def apply(X, degree=1, **kwargs):
    """
    Detrending.
    degree: 0 (constant), 1 (linear).
    """
    if degree == 1:
        type_str = 'linear'
    else:
        type_str = 'constant'
        
    return scipy_detrend(X, axis=1, type=type_str)

def transform(X_train, X_test, **kwargs):
    return apply(X_train, **kwargs), apply(X_test, **kwargs)
