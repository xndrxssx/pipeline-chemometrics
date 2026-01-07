import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_variable_importance(model, feature_names=None, model_name="", output_path=None):
    """
    Plots variable importance.
    """
    importance = None
    
    # PLS
    if hasattr(model, 'coef_') and hasattr(model, 'x_weights_'): # PLS usually
        # PLS weights (w*) or coeffs. Coeffs are better for "importance" in regression sense?
        # Often PLS weights w* or VIP scores are used.
        # Sklearn PLSRegression has `coef_` (B) and `x_weights_`.
        # Let's use coef_ absolute values or just coef_.
        importance = np.abs(model.coef_).flatten()
        
    # Tree models
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
    # Linear models (ElasticNet, SVR Linear)
    elif hasattr(model, 'coef_'):
         importance = np.abs(model.coef_).flatten()
         
    if importance is None:
        print(f"No feature importance available for model {model_name}")
        return

    # If many features (spectra), plot as line graph
    # If few features, bar chart.
    # Spectra = many features.
    
    plt.figure(figsize=(10, 5))
    
    if feature_names is None:
        feature_names = np.arange(len(importance))
        
    plt.plot(feature_names, importance, label='Importance')
    plt.fill_between(feature_names, importance, alpha=0.3)
    
    plt.xlabel('Variable / Wavelength')
    plt.ylabel('Importance')
    plt.title(f'Variable Importance: {model_name}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
