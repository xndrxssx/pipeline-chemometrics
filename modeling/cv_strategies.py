from sklearn.model_selection import KFold, LeaveOneOut

def get_kfold(n_splits=5, random_state=42):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def get_loo():
    return LeaveOneOut()
