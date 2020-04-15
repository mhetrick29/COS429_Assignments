"""
 Princeton University, COS 429, Fall 2019
"""

import numpy as np
import math 
def logistic_prob(X, params):
    """Given a logistic model and some new data, predicts probability that
       the class is 1.

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        params: vector of parameters 

    Returns:
        z: predicted probabilities (0..1)
    """
    num_pts, num_vars = X.shape
    z = np.zeros(num_pts)

    for i in range(num_pts):
         z[i] = 1 / (1+ np.exp(-(np.sum(X[i,:]*params))))

    return z
