from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class PCR(BaseEstimator, RegressorMixin):
    """
    Principal Component Regression (PCR) wrapper.
    Combines PCA and Linear Regression.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.pca_ = PCA(n_components=self.n_components)
        self.reg_ = LinearRegression()
        
        X_reduced = self.pca_.fit_transform(X)
        self.reg_.fit(X_reduced, y)
        return self

    def predict(self, X):
        X_reduced = self.pca_.transform(X)
        return self.reg_.predict(X_reduced)
