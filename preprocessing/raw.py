def transform(X_train, X_test, **kwargs):
    """
    Raw data (no processing).
    Returns copies to ensure safety.
    """
    return X_train.copy(), X_test.copy()
