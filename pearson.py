import numpy as np
from scipy.stats import pearsonr

def pearson_correlation(favorite_vector, all_vectors):
    correlations = []
    for vector in all_vectors:
        if np.array_equal(favorite_vector, vector):
            correlations.append(0)  # to avoid perfect correlation with itself
        else:
            correlation, _ = pearsonr(favorite_vector, vector)
            correlations.append(correlation)
    return np.array(correlations)