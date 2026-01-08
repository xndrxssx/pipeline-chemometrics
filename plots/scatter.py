import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def get_stats_text(y_true, y_pred, label="Set"):
    lr = LinearRegression()
    y_true_r = y_true.reshape(-1, 1)
    lr.fit(y_true_r, y_pred)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    slope = lr.coef_[0]
    offset = lr.intercept_
    bias = np.mean(y_pred - y_true)
    sd_y = np.std(y_true)
    rpd = sd_y / rmse if rmse != 0 else 0
    
    text = (f"{label} Metrics:\n"
            f"  $R^2$:  {r2:.3f}\n"
            f"  RMSE: {rmse:.3f}\n"
            f"  RPD:  {rpd:.2f}\n"
            f"  Bias: {bias:.3f}\n"
            f"  Slope:{slope:.2f}")
    return text

def plot_calibration_cv(y_train, y_pred_train, y_cv, y_pred_cv, target_name, output_path=None):
    """
    Scatter plot: Real vs Pred (Train and CV) with metrics.
    Rectangular aspect ratio.
    """
    plt.figure(figsize=(12, 6)) # Rectangular
    
    # Plot Points - Blue for Cal, Red for CV
    plt.scatter(y_train, y_pred_train, c='blue', alpha=0.5, label='Calibration')
    plt.scatter(y_cv, y_pred_cv, c='red', alpha=0.5, label='CV')
    
    # Identity line
    all_vals = np.concatenate([y_train, y_cv, y_pred_train, y_pred_cv])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    padding = (max_val - min_val) * 0.05
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Target')
    
    # Metrics Text
    txt_train = get_stats_text(y_train, y_pred_train, "Cal")
    txt_cv = get_stats_text(y_cv, y_pred_cv, "CV")
    
    # Place text boxes
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(1.02, 0.98, txt_train, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    plt.text(1.02, 0.60, txt_cv, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.xlabel(f'Measured {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'Calibration & CV: {target_name}')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim(min_val - padding, max_val + padding)
    plt.ylim(min_val - padding, max_val + padding)
    
    # Adjust layout to make room for text on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_test_prediction(y_test, y_pred_test, target_name, output_path=None, buffer=None):
    """
    Scatter plot: Real vs Pred (Test only) with detailed metrics.
    Rectangular aspect ratio.
    """
    plt.figure(figsize=(12, 6)) # Rectangular
    
    # Plot - Green points as requested
    plt.scatter(y_test, y_pred_test, c='#2ca02c', alpha=0.6, label='Test Data')
    
    # Identity line
    all_vals = np.concatenate([y_test, y_pred_test])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    padding = (max_val - min_val) * 0.05
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Target')
    
    # Fit line - Red as requested
    lr = LinearRegression()
    lr.fit(y_test.reshape(-1, 1), y_pred_test)
    line_x = np.linspace(min_val, max_val, 100).reshape(-1, 1)
    line_y = lr.predict(line_x)
    plt.plot(line_x, line_y, 'r-', alpha=0.8, lw=1.5, label='Fit Line')
    
    # Metrics
    stats_txt = get_stats_text(y_test, y_pred_test, "Test")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.text(1.02, 0.95, stats_txt, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.xlabel(f'Measured {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'External Prediction: {target_name}')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim(min_val - padding, max_val + padding)
    plt.ylim(min_val - padding, max_val + padding)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    elif buffer:
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
