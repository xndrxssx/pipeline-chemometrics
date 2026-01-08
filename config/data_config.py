from pathlib import Path

"""
Configuration module for dataset metadata.
Segunda-feira: O Aquecimento (Bioquímica)
  Datasets mais "clássicos" e fáceis de comparar.
   * bioq_folha_tellspec
   * bioq_ramo_tellspec
   * bioq_folha_fieldspec
   * bioq_ramo_fieldspec
   * Total: 4 Datasets.
   * Meta: Fechar a comparação completa Folha vs Ramo e TellSpec vs FieldSpec para
     Amido/Carboidrato.

  Terça-feira: O Pesadelo (Nutricional)
  Nutricional é pesado porque tem 10 alvos (N, P, K... Zn) por arquivo. Cada arquivo vale por
  5 dos outros.
   * nutricional_tellspec
   * nutricional_fieldspec
   * Total: 2 Datasets (mas com peso de 10).
   * Meta: Ter todos os modelos nutricionais prontos. Se sobrar tempo, adiante algo de quarta.

  Quarta-feira: A Fisiologia (IRGA)
   * fisiologia_tellspec
   * fisiologia_fieldspec
   * Total: 2 Datasets.
   * Meta: Fechar a parte de trocas gasosas (Fotossíntese, etc.). Como são dados ruidosos,
     pode demorar um pouco mais para convergir.

  Quinta-feira: O Grand Finale (Qualidade e Maturação)
  Aqui entra a classificação ("Stage") e os parâmetros de qualidade (Brix, Firmeza).
   * maturacao_tellspec
   * maturacao_fieldspec
   * exportacao_tellspec
   * exportacao_fieldspec
   * Total: 4 Datasets.
   * Meta: Fechar a parte de aplicação prática (classificação de maturação).
"""

# Root directory of the pipeline
BASE_DIR = Path(__file__).resolve().parent.parent

# Dictionary holding metadata for each dataset
# Note: wavelength_start is now detected automatically by the loader.
DATA_CONFIG = {
    # --- PRIMEIRO: BIOQUÍMICA ---
    "bioq_folha_tellspec": {
        "file": BASE_DIR / "datasets" / "bioquimica_folha_tell.xlsx",
        "targets": ["Amido Folha (mg/g)", "Carboidrato (mg/g)"],
        "label": "Bioquímica Folha (TellSpec)"
    },
    "bioq_ramo_tellspec": {
        "file": BASE_DIR / "datasets" / "bioquimica_Ramo_tell.xlsx",
        "targets": ["Amido (mg/g)", "Carboidrato (mg/g)"],
        "label": "Bioquímica Ramo (TellSpec)"
    },
    "bioq_folha_fieldspec": {
        "file": BASE_DIR / "datasets" / "Avaliacao_bioquimica_folha.xlsx",
        "targets": ["Amido Folha (mg/g)", "Carboidrato (mg/g)"],
        "label": "Bioquímica Folha (FieldSpec)"
    },
    "bioq_ramo_fieldspec": {
        "file": BASE_DIR / "datasets" / "Avaliacao_bioquimica_ramo.xlsx",
        "targets": ["Amido (mg/g)", "Carboidrato (mg/g)"],
        "label": "Bioquímica Ramo (FieldSpec)"
    },

    # --- SEGUNDO: NUTRICIONAL ---
    "nutricional_fieldspec": {
        "file": BASE_DIR / "datasets" / "Avaliacao_nutricional.xlsx",
        "targets": ["N (g/kg)", "P (g/kg)", "K (g/kg)", "Ca (g/kg)", "Mg (g/kg)", "S (g/kg)", "B (mg/kg)", "Fe (mg/kg)", "Mn (mg/kg)", "Zn (mg/kg)"],
        "label": "Status Nutricional (FieldSpec)"
    },
    "nutricional_tellspec": {
        "file": BASE_DIR / "datasets" / "nutricional_tell.xlsx",
        "targets": ["N (g/kg)", "P (g/kg)", "K (g/kg)", "Ca (g/kg)", "Mg (g/kg)", "S (g/kg)", "B (mg/kg)", "Fe (mg/kg)", "Mn (mg/kg)", "Zn (mg/kg)"],
        "label": "Status Nutricional (TellSpec)"
    },

    # --- TERCEIRO: FISIOLOGIA ---
    "fisiologia_fieldspec": {
        "file": BASE_DIR / "datasets" / "Avaliacao_fisiologica.xlsx",
        "targets": [
            "Fotossintese (μmol CO2/m²/s)", "Condutancia estomatica (mmol/m²/s¹)",
            "Carbono Interno (ppm)", "Transpiração (mol/m²/s¹)",
            "Eficiencia uso de agua (g/MJ¹)", "Eficiencia de carboxilacao (μmol CO2/m²/s)"
        ],
        "label": "Fisiologia (FieldSpec)"
    },
    "fisiologia_tellspec": {
        "file": BASE_DIR / "datasets" / "fisiologica_folha_irga.xlsx",
        "targets": [
            "Fotossintese (μmol CO2/m²/s)", "Condutancia estomatica (mmol/m²/s¹)",
            "Carbono Interno (ppm)", "Transpiração (mol/m²/s¹)",
            "Eficiencia uso de agua (g/MJ¹)", "Eficiencia de carboxilacao (μmol CO2/m²/s)"
        ],
        "label": "Fisiologia (TellSpec)"
    },

    # --- QUARTO: MATURAÇÃO (PALMER & TOMMY) ---
    "maturacao_fieldspec": {
        "file": BASE_DIR / "datasets" / "Avaliacao_Maturacao_Palmer_e_Tommy_Fieldspec.xlsx",
        "targets": ["Stage", "Firmness (N)", "Dry Mass (%)", "TSS (Brix)", "TA (g/mL)", "AA (mg/100g)"],
        "label": "Maturação Palmer/Tommy (FieldSpec)"
    },
    "maturacao_tellspec": {
        "file": BASE_DIR / "datasets" / "Avaliacao_Maturacao_Palmer_Tommy_Tellspec.xlsx",
        "targets": ["Stage", "Firmness (N)", "Dry Mass (%)", "TSS (Brix)", "TA (g/mL)", "AA (mg/100g)"],
        "label": "Maturação Palmer/Tommy (TellSpec)"
    },

    # --- QUINTO: SIMULAÇÃO EXPORTAÇÃO (KEITT & KENT) ---
    "exportacao_fieldspec": {
        "file": BASE_DIR / "datasets" / "Simulacao_Exportacao_Keitt_e_Kent_Fieldspec.xlsx",
        "targets": ["Stage", "Firmness (N)", "Dry Mass (%)", "TSS (Brix)", "TA (g/mL)", "AA (mg/100g)"],
        "label": "Exportação Keitt/Kent (FieldSpec)"
    },
    "exportacao_tellspec": {
        "file": BASE_DIR / "datasets" / "Simulacao_Exportacao_Keitt_e_Kent_Tellspec.xlsx",
        "targets": ["Stage", "Firmness (N)", "Dry Mass (%)", "TSS (Brix)", "TA (g/mL)", "AA (mg/100g)"],
        "label": "Exportação Keitt/Kent (TellSpec)"
    }
}


def get_dataset_metadata(key: str) -> dict:
    """Return metadata dictionary for the selected dataset key."""
    if key not in DATA_CONFIG:
        raise KeyError(f"Dataset '{key}' not found in config! Available: {list(DATA_CONFIG.keys())}")
    return DATA_CONFIG[key]
