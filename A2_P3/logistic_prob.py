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
    z = 1 / (1+ np.exp(-(X @ params)))

    return z
