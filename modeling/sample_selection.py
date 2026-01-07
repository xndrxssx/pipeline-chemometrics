import numpy as np
from sklearn.metrics import pairwise_distances

def kennard_stone_split(X, test_size=0.3, metric='euclidean'):
    """
    Kennard-Stone algorithm for splitting data into train/test sets based on spectral differences.
    Selects samples that cover the spectral variability uniformly.
    
    Args:
        X (np.ndarray): Spectral data matrix (samples x features).
        test_size (float): Proportion of samples to include in the test set.
        metric (str): Distance metric to use (default: 'euclidean').
        
    Returns:
        train_indices (np.ndarray): Indices for the training set.
        test_indices (np.ndarray): Indices for the test set.
    """
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Calculate distance matrix
    dists = pairwise_distances(X, metric=metric)
    
    # Select two most distant samples to start
    # We find the pair with the maximum distance
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    
    # Initialize train indices list with the two most distant points
    train_indices = [i, j]
    
    # Mask for remaining candidates
    candidates = list(set(range(n_samples)) - set(train_indices))
    
    # Iteratively select the rest of the training samples
    # We want to select n_train samples in total. We already have 2.
    for _ in range(n_train - 2):
        # For each candidate, find the minimum distance to any already selected training sample
        # We want to maximize this minimum distance (maximin criterion)
        
        # Sub-matrix of distances: rows=candidates, cols=already_selected
        sub_dists = dists[np.ix_(candidates, train_indices)]
        
        # Min distance for each candidate to the current training set
        min_dists = np.min(sub_dists, axis=1)
        
        # Select the candidate with the largest minimum distance
        best_candidate_idx = np.argmax(min_dists)
        best_candidate = candidates[best_candidate_idx]
        
        train_indices.append(best_candidate)
        candidates.pop(best_candidate_idx)
        
    # The rest go to test set
    test_indices = list(set(range(n_samples)) - set(train_indices))
    
    return np.array(train_indices), np.array(test_indices)
