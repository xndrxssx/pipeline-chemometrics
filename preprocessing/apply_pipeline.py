import importlib
import numpy as np

DEPENDENT_METHODS = ["OSC", "OPLS"]  # lista pode crescer depois

def load_method_implementation(method_name: str):
    """
    Tenta localizar o módulo do método pelo nome do arquivo em /preprocessing/
    Ex.: "SNV" → chemometrics_pipeline.preprocessing.snv
    """
    try:
        # Tenta importar usando o caminho completo do pacote
        module = importlib.import_module(f"chemometrics_pipeline.preprocessing.{method_name.lower()}")
        return module
    except ModuleNotFoundError:
        try:
            # Fallback para execução local (se rodando de dentro da pasta)
            module = importlib.import_module(f"preprocessing.{method_name.lower()}")
            return module
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Erro: método '{method_name}' não encontrado em chemometrics_pipeline/preprocessing/")


def apply_preprocessing_step(step_dict, X_train, X_test, y_train=None):
    """
    Executa uma etapa individual.
    step_dict = { "method": "SNV", "params": {...} }
    """
    method = step_dict.get("method") or step_dict.get("name")
    if not method:
        raise ValueError(f"Step dictionary missing 'method' or 'name': {step_dict}")

    params = step_dict.get("params", {})

    module = load_method_implementation(method)

    # Check case-insensitive
    if method.upper() in DEPENDENT_METHODS:
        # ANTI-LEAKAGE
        # fit somente no TREINO → depois transform no TESTE
        transformer = module.FitModel(**params)
        X_train_t = transformer.fit_transform(X_train, y_train)
        X_test_t = transformer.transform_only(X_test)
        return X_train_t, X_test_t
    else:
        # filtros independentes
        return module.transform(X_train, X_test, **params)


def apply_pipeline(methods, X_train, X_test, y_train=None, verbose=True):
    """
    Executa lista completa de preprocessing steps
    """
    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()

    for idx, step in enumerate(methods):
        method_name = step.get('method') or step.get('name')
        if verbose:
            print(f"[{idx+1}/{len(methods)}] Aplicando: {method_name} ...")

        X_train_proc, X_test_proc = apply_preprocessing_step(
            step, X_train_proc, X_test_proc, y_train
        )

    return X_train_proc, X_test_proc
