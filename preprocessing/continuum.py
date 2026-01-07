import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

def get_upper_hull(w, y):
    """
    Computes the upper convex hull of spectrum y.
    Returns the interpolated continuum.
    """
    points = np.column_stack((w, y))
    hull = ConvexHull(points)
    
    # Get vertices of the hull
    vertices = hull.vertices
    
    # We want the upper part of the hull.
    # The upper hull consists of segments connecting points that are 'above' the rest.
    # A simple heuristic: check which vertices form the upper boundary.
    # Usually, sorting vertices by x-coordinate and filtering helps.
    
    # However, standard convex hull wraps around.
    # We only want the top part connecting the start and end (mostly).
    # But usually, continuum removal uses the "rubber band" stretched over the top.
    
    # Filter vertices: In the upper hull, the y values are generally higher.
    # Actually, scipy.spatial.ConvexHull returns vertices in counter-clockwise order.
    # We can separate upper from lower by checking position relative to the line connecting first and last point.
    
    # Simplified approach for spectral continuum:
    # 1. Identify hull vertices.
    # 2. Sort them by wavelength (x).
    # 3. Interpolate between them.
    # 4. Check if the interpolation is above the signal.
    
    # Let's just use the vertices and interpolate, but we need to ensure we pick the UPPER vertices.
    
    # Sort vertices by index (wavelength position)
    vertex_indices = sorted(vertices)
    
    # The full hull is a polygon. We want the "upper" chain.
    # The upper chain starts at min(x), goes to max(x) via the top.
    
    # Let's extract points
    hull_points = points[vertices, :]
    # Sort by x
    hull_points = hull_points[np.argsort(hull_points[:, 0])]
    
    # This includes the lower part too.
    # We need to filter.
    # A vertex is on the upper hull if it's not below the segment connecting min(x) and max(x)? No.
    
    # Robust way: 
    # The upper hull is the part of the boundary accessible from y = +infinity.
    
    # Let's try:
    # 1. Start from first point.
    # 2. Find next point that maximizes slope? (Jarvis march style but for upper chain)
    # Yes. Upper chain from left to right:
    # Current point P = (w[0], y[0]).
    # Next point Q is the one that maximizes slope (y_q - y_p) / (w_q - w_p) among all future points.
    
    upper_hull_indices = [0]
    current_idx = 0
    n = len(y)
    
    while current_idx < n - 1:
        # Check all points to the right
        candidates = range(current_idx + 1, n)
        # Maximize slope
        best_slope = -np.inf
        best_idx = -1
        
        for i in candidates:
            slope = (y[i] - y[current_idx]) / (w[i] - w[current_idx])
            if slope >= best_slope:
                best_slope = slope
                best_idx = i
        
        upper_hull_indices.append(best_idx)
        current_idx = best_idx
        
    # Now we have the indices of the upper hull points
    hull_w = w[upper_hull_indices]
    hull_y = y[upper_hull_indices]
    
    # Interpolate
    f = interp1d(hull_w, hull_y, kind='linear', bounds_error=False, fill_value="extrapolate")
    return f(w)

def apply(X, method='upper', smoothing_window=5, **kwargs):
    """
    Continuum Removal.
    method: 'upper' (divides by upper convex hull)
    """
    n_samples, n_features = X.shape
    w = np.arange(n_features)
    
    X_cr = np.zeros_like(X)
    
    for i in range(n_samples):
        y = X[i, :]
        continuum = get_upper_hull(w, y)
        
        # Avoid division by zero
        continuum[continuum == 0] = 1e-10
        
        # Continuum removal: Reflected = Observed / Continuum
        X_cr[i, :] = y / continuum
        
    return X_cr

def transform(X_train, X_test, **kwargs):
    return apply(X_train, **kwargs), apply(X_test, **kwargs)
