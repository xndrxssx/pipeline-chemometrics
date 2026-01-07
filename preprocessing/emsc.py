import numpy as np

def apply(X, reference=None, include_quadratic_term=True, **kwargs):
    """
    Extended Multiplicative Scatter Correction (EMSC).
    Corrects baseline and scaling effects.
    Model: x_i = a + b*ref + c*w + d*w^2 + ...
    """
    n_samples, n_features = X.shape
    
    # Use indices as wavenumbers if not provided
    w = np.arange(n_features)
    
    # Calculate reference (mean) if not provided
    if reference is None:
        reference = np.mean(X, axis=0)
    
    # Build design matrix M for regression
    # Columns: [1, reference, w, (w^2)]
    
    # Constant term
    ones = np.ones(n_features)
    
    cols = [ones, reference, w]
    if include_quadratic_term:
        cols.append(w**2)
        
    M = np.column_stack(cols) 
    
    # Solve for coeffs: M * beta = x_i^T
    # beta = (M^T M)^-1 M^T x_i^T
    # Or using lstsq
    
    X_emsc = np.zeros_like(X)
    
    for i in range(n_samples):
        y_vec = X[i, :]
        beta, residuals, rank, s = np.linalg.lstsq(M, y_vec, rcond=None)
        
        # beta indices: 0=offset(a), 1=slope(b), 2=linear(c), 3=quad(d)
        a = beta[0]
        b = beta[1]
        c = beta[2]
        
        # Calculate correction
        # We want to remove chemical interference (a, c, d) and normalize scaling (b)
        # x_corr = (x_i - a - c*w - d*w^2) / b
        
        correction_term = a * ones + c * w
        if include_quadratic_term:
            d = beta[3]
            correction_term += d * (w**2)
            
        # Avoid division by zero
        if np.abs(b) < 1e-6:
            b = 1.0
            
        X_emsc[i, :] = (y_vec - correction_term) / b
        
    return X_emsc

def transform(X_train, X_test, **kwargs):
    # Use X_train mean as reference for both
    ref = np.mean(X_train, axis=0)
    return apply(X_train, reference=ref, **kwargs), apply(X_test, reference=ref, **kwargs)
