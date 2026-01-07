import yaml
import os
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

def load_pipeline_config():
    path = os.path.join(CONFIG_DIR, "pipeline.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError("ERRO: pipeline.yaml não encontrado em /config/")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_preprocessing_presets():
    path = os.path.join(CONFIG_DIR, "preprocessing.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError("ERRO: preprocessing.yaml não encontrado em /config/")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_selected_preset_log(preset_name, methods):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, "pipeline_used.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== Pipeline usado em {timestamp} ===\n")
        f.write(f"Preset: {preset_name}\n")
        f.write("Sequência:\n")
        for m in methods:
            f.write(f"  - {m['method']} | params={m.get('params', {})}\n")
        f.write("\n")

def load_preprocessing_steps(verbose=True, save_log=True):
    pipeline_cfg = load_pipeline_config()
    presets_cfg = load_preprocessing_presets()

    preset_name = pipeline_cfg.get("preprocessing", {}).get("preset", None)
    if preset_name is None:
        raise ValueError("ERRO: Nenhum preset definido em pipeline.yaml")

    presets_dict = presets_cfg.get("presets", {})
    if preset_name not in presets_dict:
        raise KeyError(f"ERRO: O preset '{preset_name}' não existe no preprocessing.yaml")

    methods = presets_dict[preset_name]

    if verbose:
        print("\n=== Pipeline Selecionado ===")
        print(f"Preset: {preset_name}")
        print("Sequência:")
        for m in methods:
            print(f"  → {m['method']} | params={m.get('params', {})}")

    if save_log:
        save_selected_preset_log(preset_name, methods)

    return preset_name, methods
