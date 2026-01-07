import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def calculate_metrics(y_true, y_pred):
    """
    Calculates R2, RMSE, MAE, RPD, RER, Bias, Variance, Slope, Offset, SD_Resid.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    residuals = y_true - y_pred
    
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Bias and Variance
    bias = np.mean(residuals)
    variance = np.var(residuals)
    
    # Linear Fit for Slope/Offset (Predicted vs Measured or vice versa? Usually Measured vs Predicted)
    # Here we stick to y_pred = slope * y_true + offset as a measure of fit quality relative to identity
    # OR y_true vs y_pred. The user asked for "slope and offset" in the plot which usually implies the fit line.
    # In the plot we did: fit(y_true, y_pred).
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred)
    slope = lr.coef_[0]
    offset = lr.intercept_
    
    # Residual Stats
    std_resid = np.std(residuals)
    ci95_limit = 1.96 * std_resid # Approx 95% CI margin
    
    # RPD = SD_y / RMSE
    sd_y = np.std(y_true)
    rpd = sd_y / rmse if rmse != 0 else 0
    
    # RER = Range_y / RMSE
    range_y = np.max(y_true) - np.min(y_true)
    rer = range_y / rmse if rmse != 0 else 0
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'Variance': variance,
        'RPD': rpd,
        'RER': rer,
        'Slope': slope,
        'Offset': offset,
        'SD_Resid': std_resid,
        'CI95_Limit': ci95_limit
    }
