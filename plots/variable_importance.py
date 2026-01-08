import matplotlib.pyplot as plt
import numpy as np
import os

def plot_variable_importance(model, model_name, wavelengths=None, output_path=None):
    """
    Plots variable importance (coefficients or feature importances).
    If wavelengths is provided, maps X-axis to spectral bands.
    """
    importance = None
    
    # Extract importance based on model type
    if hasattr(model, 'coef_'):
        importance = model.coef_
        # Flatten if shape is (1, n_features)
        if importance.ndim > 1:
            importance = importance.flatten()
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    if importance is None:
        print(f"No feature importance available for model {model_name}")
        return

    plt.figure(figsize=(12, 5))
    
    # X-Axis Definition
    if wavelengths is not None and len(wavelengths) == len(importance):
        x_vals = wavelengths
        x_label = "Wavelength (nm)"
    else:
        x_vals = np.arange(len(importance))
        x_label = "Variable Index"
    
    # Plotting
    plt.plot(x_vals, importance, color='black', linewidth=1)
    plt.fill_between(x_vals, 0, importance, color='gray', alpha=0.3)
    
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.xlabel(x_label)
    plt.ylabel("Coefficient / Importance")
    plt.title(f"Variable Importance: {model_name}")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
