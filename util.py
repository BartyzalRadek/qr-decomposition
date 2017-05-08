import numpy as np

def remove_target_column(matrix, ycol):
    X = np.matrix(np.zeros((matrix.shape[0], matrix.shape[1] - 1)))
    idx = 0
    for col in range(matrix.shape[1]):
        if col != ycol:
            X[:, idx] = matrix[:, col]
            idx += 1
    return X