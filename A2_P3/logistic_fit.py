"""
 Princeton University, COS 429, Fall 2019
"""
import numpy as np


def logistic_fit(X, z, l):
    """Performs L2-regularized logistic regression via Gauss-Newton iteration

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        z: labels (0/1)
        l: lambda, regularization parameter (will be scaled by the number of examples)

    Returns:
        params: vector of parameters 
    """
    num_pts, num_vars = X.shape

    # Linear regression to compute initial estimate.
    # We need to apply a correction to z for just the first
    # linear fit, since the nonlinearity isn't being applied.
    z_corr = 2 * z - 1
    params = np.linalg.inv(X.T @ X + l * num_pts * np.identity(num_vars)) @ (X.T @ z_corr)

    log = np.zeros(num_pts)
    r = np.zeros(num_pts)

# iterate to get best params
    for iter in range(10):
        # filling vector of residuals after the k-1 iteration
        prediction = logistic(X @ params)
        r = z - prediction
        # Weight Matrix "W"
        W = np.diag((prediction)*(1-prediction))  
        J = W @ X
        delta = np.linalg.inv(J.T @ J + l * num_pts * np.identity(num_vars)) @ (J.T @ r)     
        params = params + delta

    return params



def logistic(x):
    """The logistic "sigmoid" function
    """
    s = 1 / (1+ np.exp(-x))
    return s