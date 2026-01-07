import numpy as np

def apply(X, reference=None, **kwargs):
    """
    Multiplicative Scatter Correction (MSC).
    """
    if reference is None:
        reference = np.mean(X, axis=0)
    
    # X_msc = (X - a) / b
    # X = a + b * X_ref
    
    n_samples, n_features = X.shape
    X_msc = np.zeros_like(X)
    
    for i in range(n_samples):
        # Fit linear regression: X[i] vs Reference
        # polyfit(x, y, 1) returns [b, a] corresponding to y = bx + a
        # Here we regress sample (y) against reference (x) ? 
        # Usually: x_sample = a + b * x_ref
        
        fit = np.polyfit(reference, X[i, :], 1)
        b = fit[0]
        a = fit[1]
        
        X_msc[i, :] = (X[i, :] - a) / b
        
    return X_msc

def transform(X_train, X_test, **kwargs):
    # For independent application, we can use the mean of X_train as reference for both?
    # Or just apply independently (using own mean)?
    # "Independent" usually implies row-wise or independent sets.
    # However, for consistency, using X_train mean as reference for X_test is better (anti-leakage friendly).
    # But strictly following 'apply' pattern:
    
    # Strategy: Use X_train mean as the reference for both Train and Test.
    ref = np.mean(X_train, axis=0)
    
    return apply(X_train, reference=ref, **kwargs), apply(X_test, reference=ref, **kwargs)
