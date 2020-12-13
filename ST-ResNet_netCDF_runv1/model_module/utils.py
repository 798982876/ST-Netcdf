'''

This file contains helper functions for running the main neural network.
'''

import numpy as np


def batch_generator(X, y, out, batch_size):
    """
    Batch generator 
    """

    # size = X.shape[0] // batch_size*8
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    out_copy = out.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    out_copy = out_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], out_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            out_copy = out_copy[indices]
            continue