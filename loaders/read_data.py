import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Attempt to import from config. If running as a package, it might need relative or absolute paths.
try:
    from config.data_config import get_dataset_metadata
except ImportError:
    from ..config.data_config import get_dataset_metadata


@dataclass
class SpectralDataset:
    name: str
    X: np.ndarray
    y: pd.DataFrame
    wavelengths: np.ndarray
    metadata_cols: pd.DataFrame
    metadata: dict
    sensor_type: str
    raw_df: pd.DataFrame

    def describe(self):
        print(f"\n=== DATASET SUMMARY: {self.name} ===")
        print(f"Sensor       : {self.sensor_type}")
        print(f"Samples      : {self.X.shape[0]}")
        print(f"Spectra size : {self.X.shape[1]}")
        print(f"Targets      : {list(self.y.columns)}")
        print(f"Wavelengths  : {self.wavelengths[0]:.2f} → {self.wavelengths[-1]:.2f} nm")
        print(f"File loaded  : {self.metadata.get('file')}")
        print("====================================\n")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Simple wrapper to load the raw Excel dataset."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Excel file not found at: {file_path}")
    return pd.read_excel(file_path)

def identify_sensor(wavelengths: np.ndarray) -> str:
    """
    Identifies sensor type based on spectral range.
    FieldSpec: ~350-2500 nm
    TellSpec: ~900-1700 nm
    """
    start_wl = wavelengths[0]
    end_wl = wavelengths[-1]
    
    if start_wl < 400 and end_wl > 2400:
        return "FieldSpec"
    elif start_wl > 800 and end_wl < 1800:
        return "TellSpec"
    else:
        return "Unknown"

def read_spectral_dataset(key: str) -> SpectralDataset:
    """
    Load dataset using metadata configuration but detecting spectra dynamically.
    """
    metadata = get_dataset_metadata(key)
    file_path = Path(metadata["file"])

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

    df = pd.read_excel(file_path)

    # Separate spectral from non-spectral columns
    spectral_cols = []
    meta_cols = []
    
    for col in df.columns:
        try:
            # Try converting to float. If success, it's a wavelength.
            wl = float(col)
            spectral_cols.append(col)
        except ValueError:
            meta_cols.append(col)
            
    # Extract Wavelengths and X
    wavelengths = np.array(spectral_cols, dtype=float)
    X = df[spectral_cols].values.astype(float)
    
    # Identify Sensor
    sensor_type = identify_sensor(wavelengths)
    
    # Extract Targets (from config, but verified against df)
    target_names = metadata.get("targets", [])
    valid_targets = [t for t in target_names if t in df.columns]
    
    if not valid_targets:
        # If no targets found from config, try to guess or use all non-spectral?
        # For safety, strict config adherence is better, but let's warn.
        print(f"Warning: No configured targets found in {file_path}. Using all non-spectral except known metadata.")
        # Heuristic: exclude typical metadata names
        known_meta = ['sample', 'wavelength', 'variety', 'stage', 'samples', 'amostra']
        valid_targets = [c for c in meta_cols if c.lower() not in known_meta]

    y = df[valid_targets].copy()
    
    # Metadata columns (everything else)
    metadata_df = df[[c for c in meta_cols if c not in valid_targets]]

    return SpectralDataset(
        name=metadata["label"],
        X=X,
        y=y,
        wavelengths=wavelengths,
        metadata_cols=metadata_df,
        metadata=metadata,
        sensor_type=sensor_type,
        raw_df=df
    )
