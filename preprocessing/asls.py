import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _asls_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Applies Asymmetric Least Squares baseline correction to a 1D spectrum."""
    lam = float(lam)
    p = float(p)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = D.dot(D.transpose())

    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1-p) * (y < z)
    return y - z


def apply(X, lam=1e5, p=0.01, n_iter=10, **kwargs):
    """
    Apply AsLS baseline correction to spectral matrix.
    Parameters come from filters.yaml
    """
    Xc = np.zeros_like(X)
    for i in range(X.shape[0]):
        Xc[i, :] = _asls_baseline(X[i, :], lam=lam, p=p, n_iter=n_iter)
    return Xc

def transform(X_train, X_test, **kwargs):
    return apply(X_train, **kwargs), apply(X_test, **kwargs)

if __name__ == "__main__":
    X = np.random.rand(4, 200)
    X2 = apply(X)
    print(X2.shape)