import joblib
import os
import numpy as np

def save_model(model, filepath, metadata=None):
    """
    Saves the model to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # We can save a dictionary with model and metadata
    data = {'model': model}
    if metadata:
        data.update(metadata)
        
    joblib.dump(data, filepath)
    # print(f"Model saved to {filepath}")

def load_model(filepath):
    return joblib.load(filepath)
