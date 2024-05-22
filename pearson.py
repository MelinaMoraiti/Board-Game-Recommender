import numpy as np


def pearson_correlation(vector, matrix):
    # Subtract the mean of each column
    matrix_centered = matrix - np.mean(matrix, axis=0)
    vector_centered = vector - np.mean(vector)
    
    # Compute the numerator (covariance)
    numerator = np.sum(matrix_centered * vector_centered, axis=1)
    
    # Compute the denominator (product of standard deviations)
    denominator = np.sqrt(np.sum(matrix_centered**2, axis=1)) * np.sqrt(np.sum(vector_centered**2))
    
    # Avoid division by zero
    denominator[denominator == 0] = 1
    
    return numerator / denominator
