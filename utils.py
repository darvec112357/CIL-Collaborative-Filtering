import numpy as np

# UTILS
def stand_norm(A):
    """
    Normalize A by subtracting the mean and dividing by the standard deviation
    """
    col_mean = np.nanmean(A, axis=0)
    col_std = np.nanstd(A, axis=0)
    A_n = (A-col_mean)/col_std
    return A_n, col_mean, col_std

def de_norm(A, mean, std):
    """
    Reverse the standardization of A
    """
    return (A*std) + mean