import numpy as np
from scipy import stats

def calculate_confidence_interval(y_true, y_pred, confidence=0.95):
    """
    Calculates Confidence Interval for predictions based on residuals using t-distribution.
    Returns lower and upper bounds for each prediction.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    residuals = y_true - y_pred
    n = len(residuals)
    
    # Standard Error of the Estimate
    # se = sqrt(sum(residuals^2) / (n - 2))  (for simple regression)
    # Using RMSE as approximation for SE
    se = np.sqrt(np.sum(residuals**2) / (n - 1)) # Using n-1 for sample SD
    
    # t-score
    # df = n - 1 (or n - p for model params)
    # We use n - 1 for simplicity as generic approach
    t_score = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    margin_of_error = t_score * se
    
    # CI is usually around the prediction
    lower = y_pred - margin_of_error
    upper = y_pred + margin_of_error
    
    return lower, upper, margin_of_error
