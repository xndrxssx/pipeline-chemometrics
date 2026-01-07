import importlib
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV, cross_val_predict
from .cv_strategies import get_kfold, get_loo

def load_class(module_name, class_name):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return None

def train_model(X_train, y_train, model_config, cv_splits=5, random_state=42, n_jobs=10, scoring='r2'):
    """
    Trains a model using GridSearchCV (5-Fold) and evaluates with LOO.
    """
    module_name = model_config['module']
    class_name = model_config['class']
    
    ModelClass = load_class(module_name, class_name)
    if ModelClass is None:
        raise ValueError(f"Could not load model {class_name} from {module_name}")
    
    # Initialize parameters
    params = model_config.get('params', {}).copy()
    
    # Handle specific params like kernel for SVR if they are at top level
    if 'kernel' in model_config:
        params['kernel'] = model_config['kernel']
        
    # Instantiate base model
    # Note: Some models might throw error if unknown params are passed. 
    # We assume config is correct.
    model = ModelClass(**params)
    
    # Grid Search Parameters
    param_grid = model_config.get('grid', {})
    
    # CV Strategy for GridSearch (Optimization)
    cv_optim = get_kfold(n_splits=cv_splits, random_state=random_state)
    
    # Run GridSearch
    start_time = time.time()
    grid = GridSearchCV(model, param_grid, cv=cv_optim, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    best_estimator = grid.best_estimator_
    best_params = grid.best_params_
    best_score_cv5 = grid.best_score_ # This is the mean score of the best param
    
    # Now, evaluate the best estimator using LOO and also get 5-Fold predictions
    
    # 5-Fold Predictions (using cross_val_predict on the best estimator)
    # Note: This effectively retrains 5 times.
    y_pred_cv5 = cross_val_predict(best_estimator, X_train, y_train, cv=cv_optim, n_jobs=n_jobs)
    
    # LOO Predictions
    loo = get_loo()
    # LOO can be slow for large datasets. 
    # Optimization: For linear models (PLS, Ridge), LOO can be computed analytically.
    # But here we use generic cross_val_predict.
    y_pred_loo = cross_val_predict(best_estimator, X_train, y_train, cv=loo, n_jobs=n_jobs)
    
    return {
        'model': best_estimator,
        'params': best_params,
        'r2_cv5_mean': best_score_cv5,
        'y_pred_cv5': y_pred_cv5,
        'y_pred_loo': y_pred_loo,
        'cv_results': grid.cv_results_,
        'training_time': training_time
    }