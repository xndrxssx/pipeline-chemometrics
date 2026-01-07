import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the project root to sys.path if not present, to ensure absolute imports work
# This handles the case where the script is run directly or from a notebook in the root
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # loaders -> chemometrics_pipeline -> root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from chemometrics_pipeline.config.data_config import DATA_CONFIG
    from chemometrics_pipeline.loaders.read_data import load_dataset
except ImportError:
    # Fallback for when 'chemometrics_pipeline' is already in path as root (e.g. inside notebooks)
    from config.data_config import DATA_CONFIG
    from loaders.read_data import load_dataset

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_colname(name) -> str:
    """Normalize column names for safe handling, ensuring it works for numeric names too."""
    name_str = str(name)
    return (
        name_str.lower()
        .strip()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def organize_targets(save_processed=True):
    """
    Load each dataset, extract spectral data (X), targets (y) and wavelengths.
    Save standardized files if save_processed=True
    Return dict of datasets
    """

    organized = {}

    for ds_name, cfg in DATA_CONFIG.items():
        print(f"\n📌 Processing dataset: {ds_name} ...")

        df = load_dataset(cfg["file"])
        df.columns = [normalize_colname(c) for c in df.columns]

        # Detect wavelengths (NIR column names are numeric floats or starting at the threshold)
        wavelengths = [
            float(c) for c in df.columns
            if c.replace(".", "", 1).isdigit() and float(c) >= 900.577121
        ]

        if not wavelengths:
            raise ValueError(f"⚠️ Nenhuma coluna espectral encontrada no dataset {ds_name}")

        # Extract X and y
        X = df[wavelengths].values.astype(np.float32)

        target_cols = [normalize_colname(col) for col in cfg["targets"]]
        y = df[target_cols].copy()

        # Store structured
        organized[ds_name] = {
            "X": X,
            "y": y,
            "target_names": target_cols,
            "wavelengths": np.array(wavelengths),
        }

        print(f"✔ X shape = {X.shape}")
        print(f"✔ y shape = {y.shape}")
        print(f"🎯 Targets = {target_cols[:4]} ... ({len(target_cols)} total)")
        print(f"📡 Wavelengths range = {min(wavelengths)} – {max(wavelengths)} nm")

        # Save to disk
        if save_processed:
            np.save(OUTPUT_DIR / f"{ds_name}_X.npy", X)
            y.to_csv(OUTPUT_DIR / f"{ds_name}_y.csv", index=False)

            meta = {
                "dataset": ds_name,
                "file": cfg["file"],
                "targets": target_cols,
                "wavelengths": wavelengths,
                "n_samples": len(df),
                "n_wavelengths": len(wavelengths),
            }
            with open(OUTPUT_DIR / f"{ds_name}_meta.json", "w") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

            print(f"💾 Saved processed files for {ds_name}")

    print("\n🎉 Finished organizing datasets.")
    return organized


if __name__ == "__main__":
    organize_targets()
