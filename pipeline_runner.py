import os
import sys
import warnings
from pathlib import Path

# --- ABSOLUTE FIRST STEP: Fix joblib resource_tracker on Windows ---
# 1. Set a persistent temp folder
temp_folder = Path("temp_joblib")
temp_folder.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(temp_folder.absolute())

# 2. Aggressively silence resource_tracker warnings at the source
def warn_ignore_resource_tracker(message, category, filename, lineno, file=None, line=None):
    if "resource_tracker" in str(message) or "joblib" in str(filename):
        return
    # Default behavior for other warnings
    print(f"{filename}:{lineno}: {category.__name__}: {message}", file=sys.stderr)

# Patching the warning system
warnings.showwarning = warn_ignore_resource_tracker
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

print("⏳ Carregando bibliotecas...", flush=True)

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib
import io
import base64
import matplotlib.pyplot as plt

# Force matplotlib to use non-interactive backend 'Agg' before importing pyplot
matplotlib.use('Agg')

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')
# Suppress joblib resource_tracker warnings (noisy on Windows)
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from config.data_config import DATA_CONFIG, get_dataset_metadata
from loaders.read_data import read_spectral_dataset
from diagnostics.pca_diagnostics import pca_diagnostics
from diagnostics.target_outliers import analyze_targets
from preprocessing.apply_pipeline import apply_pipeline
from modeling.train_model import train_model
from modeling.save_best import save_model
from stats_utils.metrics import calculate_metrics
from stats_utils.confidence_interval import calculate_confidence_interval
from plots.boxplot_target import plot_target_boxplot
from plots.scatter import plot_calibration_cv, plot_test_prediction
from plots.variable_importance import plot_variable_importance
from utils.html_generator import generate_interactive_report

print("Bibliotecas carregadas!", flush=True)

from modeling.sample_selection import kennard_stone_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class PipelineRunner:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.config_dir = self.base_dir / "config"
        self.output_dir = self.base_dir / "outputs"
        self.models_config = self._load_yaml(self.config_dir / "models.yaml")
        self.filters_config = self._load_yaml(self.config_dir / "filters.yaml")
        
        # Results storage
        self.results = []
        
    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def generate_preprocessing_grid(self):
        """
        Generates a list of preprocessing chains to test.
        Combines:
        1. Presets defined in config/preprocessing.yaml
        2. Dynamic Grids defined in config/filters.yaml (SG windows, AsLS lambdas, etc.)
        """
        chains = []
        
        # 1. Load Presets from preprocessing.yaml
        try:
            with open(self.config_dir / "preprocessing.yaml", 'r') as f:
                prep_config = yaml.safe_load(f)
            if prep_config and 'presets' in prep_config:
                # print(f"      [Config] Loading {len(prep_config['presets'])} presets from preprocessing.yaml...")
                for name, chain in prep_config['presets'].items():
                    chains.append(chain)
        except Exception as e:
            print(f"      [Config] Warning: Could not load preprocessing.yaml: {e}")

        # 2. Generate Dynamic Grid from filters.yaml
        try:
            filters_cfg = self.filters_config.get('preprocessing', {})
            y_dep_cfg = self.filters_config.get('y_dependent', {})

            # RAW
            if filters_cfg.get('raw', {}).get('enabled', True):
                 if not any(c == [{"method": "raw"}] for c in chains):
                    chains.append([{"method": "raw"}])

            # SG (Savitzky-Golay)
            sg_cfg = filters_cfg.get('sg', {})
            if sg_cfg.get('enabled', False):
                for w in sg_cfg.get('windows', [11]):
                    for p in sg_cfg.get('polyorders', [2]):
                        for d in sg_cfg.get('derivatives', [1]):
                            if w > p: # Valid constraint
                                chains.append([{"method": "sg", "params": {"window_length": w, "polyorder": p, "deriv": d}}])

            # AsLS
            asls_cfg = filters_cfg.get('asls', {})
            if asls_cfg.get('enabled', False):
                for lam in asls_cfg.get('lambda', [1e4]):
                    chains.append([{"method": "asls", "params": {"lam": lam, "p": asls_cfg.get('p', [0.05])[0]}}])

            # SNV
            if filters_cfg.get('snv', {}).get('enabled', False):
                chains.append([{"method": "snv"}])

            # MSC
            if filters_cfg.get('msc', {}).get('enabled', False):
                chains.append([{"method": "msc"}])

            # Detrend
            det_cfg = filters_cfg.get('detrend', {})
            if det_cfg.get('enabled', False):
                for deg in det_cfg.get('degree', [1]):
                    chains.append([{"method": "detrend", "params": {"degree": deg}}])
            
            # Normalize
            norm_cfg = filters_cfg.get('normalize', {})
            if norm_cfg.get('enabled', False):
                for mode in norm_cfg.get('mode', ['l2']):
                    chains.append([{"method": "normalize", "params": {"mode": mode}}])
                    
            # Continuum Removal
            cont_cfg = filters_cfg.get('continuum', {})
            if cont_cfg.get('enabled', False):
                chains.append([{"method": "continuum"}])
                
            # EMSC
            emsc_cfg = filters_cfg.get('emsc', {})
            if emsc_cfg.get('enabled', False):
                chains.append([{"method": "emsc", "params": {"include_quadratic_term": emsc_cfg.get('include_quadratic_term', True)}}])

            # Wavelet
            wav_cfg = filters_cfg.get('wavelet', {})
            if wav_cfg.get('enabled', False):
                for w_name in wav_cfg.get('wavelets', ['db4']):
                    for lvl in wav_cfg.get('levels', [1]):
                        chains.append([{"method": "wavelet", "params": {"wavelet": w_name, "level": lvl}}])

            # Anti-Leakage (OSC/OPLS) defined in filters.yaml
            osc_cfg = y_dep_cfg.get('osc', {})
            if osc_cfg.get('enabled', False):
                for n in osc_cfg.get('components', [1]):
                    chains.append([{"method": "osc", "params": {"components": n}}])

            opls_cfg = y_dep_cfg.get('opls', {})
            if opls_cfg.get('enabled', False):
                for n in opls_cfg.get('max_components', [1]):
                    chains.append([{"method": "opls", "params": {"max_components": n}}])

        except Exception as e:
            print(f"      [Config] Warning: Error parsing filters.yaml for grid generation: {e}")
            # Fallback to minimal if fails
            chains.append([{"method": "raw"}])
            
        return chains
        
        # Detrend (Linear & Constant)
        chains.append([{"method": "detrend", "params": {"degree": 1}}])
        
        # Normalize (L2)
        chains.append([{"method": "normalize", "params": {"mode": "l2"}}])
        
        # Continuum Removal
        chains.append([{"method": "continuum"}])
        
        # Wavelet Denoising (Added)
        # Note: These parameters mirror what is in config/filters.yaml
        for w_name in ['db4', 'sym8']:
            for lvl in [1, 2]:
                chains.append([{"method": "wavelet", "params": {"wavelet": w_name, "level": lvl}}])
        
        # 3. Combinations (Common in Chemometrics)
        # SNV + SG (D1) - Using middle window
        chains.append([
            {"method": "snv"},
            {"method": "sg", "params": {"window_length": 11, "polyorder": 2, "deriv": 1}}
        ])
        
        # MSC + SG (D1)
        chains.append([
            {"method": "msc"},
            {"method": "sg", "params": {"window_length": 11, "polyorder": 2, "deriv": 1}}
        ])

        # 4. Anti-Leakage (Fit on Train)
        # OSC (1, 2, 3 components)
        for n_comp in [1, 2, 3]:
            chains.append([{"method": "osc", "params": {"components": n_comp}}])
            
        # OPLS (1, 2, 3 components)
        for n_comp in [1, 2, 3]:
            chains.append([{"method": "opls", "params": {"max_components": n_comp}}])
            
        return chains

    def _get_chain_name(self, chain):
        """Generates a string representation of the preprocessing chain."""
        names = []
        for step in chain:
            name = step.get('method') or step.get('name')
            if not name:
                continue
                
            if 'params' in step:
                # Add key params for differentiation
                if name == 'sg':
                    name += f"_w{step['params'].get('window_length')}_d{step['params'].get('deriv')}"
                elif name == 'asls':
                    name += f"_lam{step['params'].get('lam')}"
                elif name in ['osc', 'opls']:
                    name += f"_comp{step['params'].get('components') or step['params'].get('max_components')}"
            names.append(name)
        return "+".join(names)

    def get_variable_selection_window(self, target_name):
        """
        Returns a list of (start, end) tuples for wavelength selection based on literature.
        Returns None if no specific window is defined (Full Spectrum).
        """
        t = target_name.lower()
        
        # Bioquímica
        if 'amido' in t or 'carboidrato' in t:
            return [(1100, 1300), (1450, 1550)]
            
        # Fisiologia (Fotossíntese, Condutância, Transpiração)
        if any(x in t for x in ['fotossintese', 'condutancia', 'transpiração', 'transpiracao', 'eficiencia uso de agua']):
            # Literature points: 970, 1200, 1450. Using +/- 10nm windows.
            return [(960, 980), (1190, 1210), (1440, 1460)]
            
        if 'carboxilacao' in t or 'carboxilação' in t:
            return [(700, 750)]
            
        # Qualidade
        if 'materia seca' in t or 'dm' in t or 'tss' in t or 'brix' in t:
            return [(900, 1100)]
            
        if 'firmeza' in t:
            return [(800, 1100)]
            
        if 'acidez' in t or 'ta' in t or 'vitamina' in t or 'aa' in t:
            return [(700, 950)]
            
        # Nutricional
        if 'n (' in t or t == 'n': # Nitrogen
            return [(1500, 1520), (2170, 2190)]
            
        # Default (Full Spectrum)
        return None

    def _slice_wavelengths(self, X, wavelengths, ranges):
        """
        Slices spectral matrix X to keep only columns within specified wavelength ranges.
        ranges: list of tuples [(start, end), ...] or single tuple (start, end)
        """
        if ranges is None:
            return X, wavelengths
            
        if isinstance(ranges, tuple):
            ranges = [ranges]
            
        mask = np.zeros(len(wavelengths), dtype=bool)
        for (start, end) in ranges:
            mask |= ((wavelengths >= start) & (wavelengths <= end))
            
        return X[:, mask], wavelengths[mask]

    def get_debug_grid(self):
        """Returns a minimal grid for quick testing."""
        print("⚡ DEBUG MODE ACTIVE: Running minimal grid.")
        return [
            [{"method": "raw"}],
            [{"method": "sg", "params": {"window_length": 15, "polyorder": 2, "deriv": 1}}]
        ]

    def run(self):
        print("Starting Chemometrics Pipeline...")
        
        # 1. Iterate Datasets (Prioritize FieldSpec)
        dataset_keys = list(DATA_CONFIG.keys())
        dataset_keys.sort(key=lambda k: 0 if "fieldspec" in k.lower() else 1)
        
        for dataset_key in dataset_keys:
            print(f"\n\nLoading Dataset: {DATA_CONFIG[dataset_key]['label']} ({dataset_key})")
            
            try:
                ds = read_spectral_dataset(dataset_key)
            except Exception as e:
                print(f"Failed to load {dataset_key}: {e}")
                continue
            
            # 2. Global Diagnostics (Spectra)
            print(f"   Sensor Detected: {ds.sensor_type}")
            print("   🔍 Running Spectral Diagnostics...")
            pca_diagnostics(ds.X, dataset_name=dataset_key, save_plots=True)
            
            # 3. Iterate Targets
            for target_col in ds.y.columns:
                print(f"\n   Processing Target: {target_col}")
                
                # Target Diagnostics (Outliers)
                y = ds.y[target_col].dropna()
                if len(y) < 10:
                    print(f"      Not enough samples for {target_col}, skipping.")
                    continue
                
                # Run Target Outlier Analysis
                analyze_targets(dataset_key, ds.y[[target_col]])
                    
                # Align X with valid y
                X_aligned = ds.X[y.index]
                y_aligned = y.values
                
                # 4. Split Data (Kennard-Stone)
                test_size = self.models_config['global']['test_size']
                try:
                    train_idx, test_idx = kennard_stone_split(X_aligned, test_size=test_size, metric='euclidean')
                    X_train_orig, X_test_orig = X_aligned[train_idx], X_aligned[test_idx]
                    y_train, y_test = y_aligned[train_idx], y_aligned[test_idx]
                except Exception as e:
                    print(f"      Split Error: {e}")
                    continue
                
                print(f"      Start Training Loop (Train={len(y_train)}, Test={len(y_test)})")
                
                # Define Conditions based on Sensor
                conditions = []
                if ds.sensor_type == "FieldSpec":
                    conditions.append(("Full", (350, 2500)))
                    conditions.append(("TellSpec_Sim", (900, 1700)))
                else:
                    # TellSpec or Unknown -> Use available full range
                    conditions.append(("Full", (ds.wavelengths[0], ds.wavelengths[-1])))
                
                for cond_name, wl_ranges in conditions:
                    print(f"      ► Condition: {cond_name} {wl_ranges}")
                    
                    # Apply Spectral Slicing
                    X_train, wl_train = self._slice_wavelengths(X_train_orig, ds.wavelengths, wl_ranges)
                    X_test, wl_test = self._slice_wavelengths(X_test_orig, ds.wavelengths, wl_ranges)
                    
                    if X_train.shape[1] == 0:
                        print(f"         ⚠ No wavelengths remaining for {cond_name}. Skipping.")
                        continue

                    # Track best model for this target/condition
                    best_r2_test = -np.inf
                    best_artifact = None
                    
                    # Determine Task Type
                    task_type = "Classification" if "stage" in target_col.lower() else "Regression"
                    scoring_metric = "accuracy" if task_type == "Classification" else "r2"
                    
                    # Prepare Y for Classification
                    y_train_curr, y_test_curr = y_train, y_test
                    if task_type == "Classification":
                        le = LabelEncoder()
                        y_train_curr = le.fit_transform(y_train)
                        y_test_curr = le.transform(y_test)

                    # 5. Iterate Preprocessing (Full Exhaustive Grid)
                    prep_grid = self.generate_preprocessing_grid()
                    # prep_grid = self.get_debug_grid() # <--- DEBUG MODE OFF
                    
                    for prep_chain in tqdm(prep_grid, desc=f"         Preprocessing ({cond_name})", leave=False):
                        prep_name = self._get_chain_name(prep_chain)
                        
                        try:
                            # Apply Preprocessing (Anti-Leakage enforced in apply_pipeline)
                            X_train_proc, X_test_proc = apply_pipeline(
                                prep_chain, X_train, X_test, y_train, verbose=False
                            )
                            
                            # 6. Iterate Models
                            for model_key, model_cfg in self.models_config['models'].items():
                                
                                # --- DEBUG FILTER: Only run PLS (REMOVED) ---
                                # if "pls" not in model_key and "pcr" not in model_key:
                                #    continue
                                # ----------------------------------

                                # Filter Models by Task
                                model_type = model_cfg.get('type', 'regression')
                                if task_type == "Regression" and model_type == "classification":
                                    continue
                                if task_type == "Classification" and model_type != "classification":
                                    continue
                                    
                                if not model_cfg.get('enabled', False):
                                    if task_type == "Classification" and model_type == "classification":
                                        pass
                                    else:
                                        continue
                                
                                # Train (GridSearch + CV)
                                try:
                                    train_res = train_model(
                                        X_train_proc, y_train_curr, model_cfg, 
                                        cv_splits=self.models_config['global']['cv']['kfold'],
                                        n_jobs=self.models_config['global']['n_jobs'],
                                        scoring=scoring_metric
                                    )
                                    
                                    # Evaluate on Test
                                    y_pred_test = train_res['model'].predict(X_test_proc)
                                    
                                    # Calculate Metrics
                                    metrics = {}
                                    if task_type == "Regression":
                                        metrics = calculate_metrics(y_test_curr, y_pred_test)
                                        primary_score = metrics['R2']
                                        r2_diff = train_res['r2_cv5_mean'] - metrics['R2']
                                    else:
                                        # Classification Metrics
                                        if "PLS" in str(train_res['model']):
                                            y_pred_test = np.round(y_pred_test).astype(int)
                                            y_pred_test = np.clip(y_pred_test, 0, len(le.classes_)-1)
                                        
                                        acc = accuracy_score(y_test_curr, y_pred_test)
                                        f1 = f1_score(y_test_curr, y_pred_test, average='weighted')
                                        prec = precision_score(y_test_curr, y_pred_test, average='weighted', zero_division=0)
                                        rec = recall_score(y_test_curr, y_pred_test, average='weighted', zero_division=0)
                                        
                                        metrics = {
                                            'Accuracy': acc,
                                            'F1_Score': f1,
                                            'Precision': prec,
                                            'Recall': rec
                                        }
                                        primary_score = acc
                                        r2_diff = 0
                                    
                                    # Generate In-Memory Plot for Report
                                    plot_b64 = None
                                    try:
                                        buf = io.BytesIO()
                                        if task_type == "Regression":
                                            plot_test_prediction(y_test_curr, y_pred_test, target_col, buffer=buf)
                                        else:
                                            # Placeholder for classification plot (could be confusion matrix)
                                            pass
                                        
                                        if buf.getbuffer().nbytes > 0:
                                            plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                        buf.close()
                                    except Exception as e:
                                        # print(f"Plot gen error: {e}")
                                        pass

                                    # Store result
                                    result_row = {
                                        "dataset": dataset_key,
                                        "target": target_col,
                                        "Sensor": ds.sensor_type,
                                        "Condition": cond_name,
                                        "Spectral_Range": str(wl_ranges),
                                        "Task": task_type,
                                        "preprocessing": prep_name,
                                        "model": model_key,
                                        "params": str(train_res['params']),
                                        "Training_Time": train_res.get('training_time'),
                                        "Plot_Base64": plot_b64,
                                        
                                        # Regression Metrics (None for Classif)
                                        "Target_SD": np.std(y_aligned) if task_type == "Regression" else None,
                                        "R2_CV": train_res['r2_cv5_mean'] if task_type == "Regression" else None,
                                        "R2_Test": metrics.get('R2'),
                                        "R2_Diff": r2_diff if task_type == "Regression" else None,
                                        "RMSE_Test": metrics.get('RMSE'),
                                        "RPD": metrics.get('RPD'),
                                        "Bias": metrics.get('Bias'),
                                        "Slope": metrics.get('Slope'),
                                        "Offset": metrics.get('Offset'),
                                        "SD_Resid": metrics.get('SD_Resid'),
                                        "CI95_Limit": metrics.get('CI95_Limit'),
                                        
                                        # Classification Metrics (None for Reg)
                                        "Accuracy": metrics.get('Accuracy'),
                                        "F1_Score": metrics.get('F1_Score'),
                                        "Precision": metrics.get('Precision'),
                                        "Recall": metrics.get('Recall')
                                    }
                                    self.results.append(result_row)
                                    
                                    # Check if best
                                    if primary_score > best_r2_test:
                                        best_r2_test = primary_score
                                        best_artifact = {
                                            "row": result_row,
                                            "model": train_res['model'],
                                            "X_train": X_train_proc, 
                                            "y_train": y_train_curr,
                                            "y_pred_cv": train_res['y_pred_cv5'],
                                            "X_test": X_test_proc,
                                            "y_test": y_test_curr,
                                            "y_pred_test": y_pred_test,
                                            "wavelengths": wl_train # STORE WAVELENGTHS HERE
                                        }
                                
                                except Exception as e:
                                    # print(f"Error training {model_key}: {e}")
                                    pass
                                    
                        except Exception as e:
                            print(f"      Error in preprocessing {prep_name}: {e}")
                            continue
                    
                    # 7. Finalize Target (Save Best and Target Report)
                    if best_artifact:
                        # Append Condition to prefix to allow multiple best models (Full vs Trimmed)
                        self._save_best_target_results(dataset_key, target_col, best_artifact, suffix=f"_{cond_name}")
                        print(f"         🏆 Best ({cond_name}): {best_artifact['row']['model']} (R2: {best_r2_test:.4f})")
                    else:
                        print(f"         No successful model found for {cond_name}.")
                
                # Save report for this target (all conditions)
                self._save_target_report(dataset_key, target_col, ds.sensor_type)
                    
        # 8. Save final consolidated results
        self._save_summary()

    def _save_target_report(self, dataset, target, sensor):
        """Saves all trials for a specific target to an Excel file."""
        target_results = [r for r in self.results if r['dataset'] == dataset and r['target'] == target]
        if not target_results:
            return
            
        df = pd.DataFrame(target_results)
        sanitized_target = target.replace("/", "_").replace(" ", "_")
        report_path = self.output_dir / "reports" / "targets" / f"{dataset}_{sanitized_target}_{sensor}_all_trials.xlsx"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure R2_Test is sortable (fill None with -inf for sorting purposes if needed, but standard sort usually handles it)
        df.sort_values(by="R2_Test", ascending=False, inplace=True) 
        df.to_excel(report_path, index=False)

    def _save_best_target_results(self, dataset, target, artifact, suffix=""):
        """Generates plots and saves model for the winning combination."""
        sanitized_target = target.replace("/", "_").replace(" ", "_")
        # Ensure suffix is present to avoid generic overwrites, or explicit
        prefix = f"{dataset}_{sanitized_target}{suffix}"
        
        # 1. Save Model
        model_path = self.output_dir / "models" / f"{prefix}_best_model.joblib"
        save_model(artifact['model'], model_path, metadata=artifact['row'])
        
        # 2. Plots
        # Boxplot
        plot_target_boxplot(artifact['y_train'], artifact['y_test'], target, 
                           output_path=self.output_dir / "plots" / "targets" / f"{prefix}_boxplot.png")
                           
        # Calibration / CV
        plot_calibration_cv(artifact['y_train'], 
                            artifact['model'].predict(artifact['X_train']),
                            artifact['y_train'], 
                            artifact['y_pred_cv'], 
                            target,
                            output_path=self.output_dir / "plots" / "calibration" / f"{prefix}_cal_cv.png")
                            
        # Test Prediction
        plot_test_prediction(artifact['y_test'], artifact['y_pred_test'], target,
                             output_path=self.output_dir / "plots" / "test" / f"{prefix}_test.png")
                             
        # Variable Importance (Pass Wavelengths)
        # We need to reconstruct wavelengths from X_train_proc if possible, 
        # but X_train_proc might be transformed (PCA scores, etc). 
        # For simple filters (SG, MSC), column count matches.
        # If mismatch, plot by index.
        wl_to_plot = None
        # We assume artifact['X_train'] has correct shape matching the model input
        # If we had the original sliced wavelengths stored, we could use them.
        # Let's try to infer if we saved them in artifact or passed them.
        # For now, we will use index if transformed.
        
        # Improvement: Pass current range wavelengths if no dimension reduction happened
        # Ideally, we should store used_wavelengths in artifact.
        
        plot_variable_importance(artifact['model'], 
                                 model_name=artifact['row']['model'],
                                 wavelengths=artifact.get('wavelengths'), # Need to add this to artifact creation
                                 output_path=self.output_dir / "plots" / "importance" / f"{prefix}_importance.png")
                                 
        # Confidence Intervals
        _, _, margin = calculate_confidence_interval(artifact['y_test'], artifact['y_pred_test'])

    def _save_summary(self):
        if not self.results:
            print("No results to save.")
            return
            
        df = pd.DataFrame(self.results)
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Regression Summary ---
        df_reg = df[df['Task'] == 'Regression']
        if not df_reg.empty:
            # Clean Columns (Drop Classification Metrics which are all NaN)
            df_reg_clean = df_reg.dropna(axis=1, how='all')
            
            # Full Log
            log_path = reports_dir / "final_results_regression.xlsx"
            df_reg_clean.sort_values(by=["dataset", "target", "R2_Test"], ascending=[True, True, False], inplace=True)
            df_reg_clean.to_excel(log_path, index=False)
            
            # Best Summary
            best_df = df_reg_clean.sort_values("R2_Test", ascending=False).drop_duplicates(["dataset", "target", "Condition"])
            best_df.sort_values(by=["dataset", "target", "Condition"], inplace=True)
            best_path = reports_dir / "best_models_regression.xlsx"
            best_df.to_excel(best_path, index=False)
            
            print(f"\n📄 Regression Log: {log_path}")
            print(f"🏆 Regression Best: {best_path}")

        # --- Classification Summary ---
        df_cls = df[df['Task'] == 'Classification']
        if not df_cls.empty:
            # Clean Columns
            df_cls_clean = df_cls.dropna(axis=1, how='all')
            
            # Full Log
            log_path = reports_dir / "final_results_classification.xlsx"
            df_cls_clean.sort_values(by=["dataset", "target", "Accuracy"], ascending=[True, True, False], inplace=True)
            df_cls_clean.to_excel(log_path, index=False)
            
            # Best Summary
            best_df = df_cls_clean.sort_values("Accuracy", ascending=False).drop_duplicates(["dataset", "target", "Condition"])
            best_df.sort_values(by=["dataset", "target", "Condition"], inplace=True)
            best_path = reports_dir / "best_models_classification.xlsx"
            best_df.to_excel(best_path, index=False)
            
            print(f"\n📄 Classification Log: {log_path}")
            print(f"🏆 Classification Best: {best_path}")

        # --- HTML Interactive Report ---
        try:
            html_path = reports_dir / "chemometrics_report.html"
            generate_interactive_report(self.results, html_path)
        except Exception as e:
            print(f"Error generating HTML report: {e}")

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
