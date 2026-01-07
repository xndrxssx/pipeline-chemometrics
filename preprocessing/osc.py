import numpy as np

class FitModel:
    def __init__(self, components=1, remove_mean=True, **kwargs):
        self.n_components = components
        self.remove_mean = remove_mean
        self.w_ortho_list = []
        self.p_ortho_list = []
        self.mean_X = None
        self.mean_y = None

    def fit_transform(self, X, y):
        """
        Fits OSC model and returns corrected X.
        Reference: Wold et al. 1998, Orthogonal Signal Correction.
        """
        X = np.array(X)
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Center data
        if self.remove_mean:
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

        # Loop for components
        for _ in range(self.n_components):
            # 1. Calculate PC1 of X (start guess)
            # Use NIPALS or SVD for first component
            u, s, vt = np.linalg.svd(X_curr, full_matrices=False)
            t = u[:, 0] * s[0] # Score
            
            # Ensure t is orthogonal to Y
            # t_new = t - Y (Y'Y)^-1 Y' t
            # For numerical stability with potential singular Y'Y, use lstsq
            
            # Project t onto Y
            # t_y = Y * beta
            beta, _, _, _ = np.linalg.lstsq(y_curr, t, rcond=None)
            t_y = y_curr @ beta
            
            # Orthogonalize
            t_ortho = t - t_y
            
            # Check if t_ortho is significant
            if np.linalg.norm(t_ortho) < 1e-10:
                break
                
            # Update weights w = X^T t_ortho / ||t_ortho||^2 ? 
            # Actually standard OSC iterates here to find a weight vector w such that
            # Xw is orthogonal to Y.
            
            # Wold's algorithm iteration:
            # 1. t = X w (start with PC1 w)
            # 2. t_ortho = (I - Y(Y'Y)^-1 Y') t
            # 3. w = X' t_ortho
            # 4. w = w / ||w||
            # 5. t = X w
            # Repeat until w converges.
            
            # Let's implement the iteration
            w = vt[0, :].T # Initial weight from PC1
            
            for iter_idx in range(10): # Max iterations
                t = X_curr @ w
                
                # Orthogonalize t w.r.t Y
                beta_y, _, _, _ = np.linalg.lstsq(y_curr, t, rcond=None)
                t_y = y_curr @ beta_y
                t_ortho = t - t_y
                
                # Update w
                w_new = X_curr.T @ t_ortho
                w_norm = np.linalg.norm(w_new)
                if w_norm == 0:
                    break
                w_new /= w_norm
                
                # Check convergence
                if np.allclose(w, w_new, atol=1e-6):
                    w = w_new
                    break
                w = w_new
            
            # Final t_ortho calculation with converged w
            t = X_curr @ w
            
            # Orthogonalize again to be sure
            beta_y, _, _, _ = np.linalg.lstsq(y_curr, t, rcond=None)
            t_ortho = t - (y_curr @ beta_y)
            
            # Calculate loading p
            # p = X' t_ortho / (t_ortho' t_ortho)
            p_num = X_curr.T @ t_ortho
            p_den = t_ortho.T @ t_ortho
            p = p_num / p_den
            
            # Store w and p
            self.w_ortho_list.append(w)
            self.p_ortho_list.append(p)
            
            # Deflate X
            # X_new = X - t_ortho * p'
            X_curr = X_curr - np.outer(t_ortho, p)
            
        return X_curr + (self.mean_X if self.remove_mean else 0)

    def transform_only(self, X):
        """
        Applies learned OSC to new data.
        """
        if self.remove_mean:
            X_curr = X - self.mean_X
        else:
            X_curr = X.copy()
            
        for w, p in zip(self.w_ortho_list, self.p_ortho_list):
            # Calculate t using the weight w
            t = X_curr @ w
            
            # We assume this t is the orthogonal score. 
            # In 'fit', we forced t to be orthogonal to Y. 
            # Here Y is unknown (test set), so we just project using w.
            # The assumption is that w captures the orthogonal direction.
            
            # Subtract
            X_curr = X_curr - np.outer(t, p)
            
        return X_curr + (self.mean_X if self.remove_mean else 0)
