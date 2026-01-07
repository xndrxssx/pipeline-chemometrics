import numpy as np

class FitModel:
    def __init__(self, max_components=1, scaling='center', **kwargs):
        self.n_components = max_components
        self.scaling = scaling
        self.w_ortho_list = []
        self.p_ortho_list = []
        self.mean_X = None
        self.mean_y = None

    def fit_transform(self, X, y):
        """
        Applies O-PLS filter.
        Removes components from X that are orthogonal to Y.
        """
        X = np.array(X)
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Center data
        if self.scaling == 'center':
            self.mean_X = np.mean(X, axis=0)
            self.mean_y = np.mean(y, axis=0)
            X_curr = X - self.mean_X
            y_curr = y - self.mean_y
        else:
            self.mean_X = np.zeros(X.shape[1])
            X_curr = X.copy()
            y_curr = y.copy()

        self.w_ortho_list = []
        self.p_ortho_list = []
        
        # O-PLS Algorithm (Trygg & Wold)
        
        for _ in range(self.n_components):
            # 1. Calculate weight w vector
            # w = X' y / y' y
            # Avoid explicit matrix inversion for y'y scalar
            
            # For multi-y, we normally pick the first component or SVD. 
            # Assuming single y here mostly.
            
            w = X_curr.T @ y_curr
            w /= np.linalg.norm(w)
            
            # 2. Calculate score t
            t = X_curr @ w
            t_denom = t.T @ t
            if t_denom == 0: break
            
            # 3. Calculate loading p
            p = X_curr.T @ t / t_denom
            
            # 4. Orthogonal weight w_ortho
            # w_ortho = p - ((w' p) / (w' w)) * w
            # Since w is normalized, w'w = 1
            w_dot_p = w.T @ p
            w_ortho = p - w_dot_p * w
            
            w_ortho_norm = np.linalg.norm(w_ortho)
            
            # If w_ortho is small, no more orthogonal variation
            if w_ortho_norm < 1e-10:
                break
                
            w_ortho /= w_ortho_norm
            
            # 5. Calculate orthogonal score t_ortho
            t_ortho = X_curr @ w_ortho
            t_ortho_denom = t_ortho.T @ t_ortho
            if t_ortho_denom == 0: break
            
            # 6. Calculate orthogonal loading p_ortho
            p_ortho = X_curr.T @ t_ortho / t_ortho_denom
            
            # Store
            self.w_ortho_list.append(w_ortho)
            self.p_ortho_list.append(p_ortho)
            
            # 7. Remove orthogonal component (Deflation)
            X_curr = X_curr - np.outer(t_ortho, p_ortho)
            
        return X_curr + (self.mean_X if self.scaling == 'center' else 0)

    def transform_only(self, X):
        """
        Applies O-PLS filter to new data.
        """
        if self.scaling == 'center':
            X_curr = X - self.mean_X
        else:
            X_curr = X.copy()
            
        for w_ortho, p_ortho in zip(self.w_ortho_list, self.p_ortho_list):
            # Calculate t_ortho
            t_ortho = X_curr @ w_ortho
            
            # Remove
            X_curr = X_curr - np.outer(t_ortho, p_ortho)
            
        return X_curr + (self.mean_X if self.scaling == 'center' else 0)
