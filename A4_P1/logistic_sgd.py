"""
 Princeton University, COS 429, Fall 2019
"""
import numpy as np


def logistic_sgd(X, z, num_epochs):
    """Performs L2-regularized logistic regression via Stochastic Gradient Descent 

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        z: labels (0/1)
        num_epochs: number of epochs to train over

    Returns:
        params: vector of parameters 
    """
    num_pts, num_vars = X.shape

    # Initial (random) estimate of params.
    mean = 0
    sigma = 1 / np.sqrt(num_vars / 2)
    params = np.random.normal(mean, sigma, num_vars) # this is w(0)

    # Loop over epochs
    for ep in range(1, num_epochs+1):
        # Permute the data rows
        permutation = np.random.permutation(num_pts)
        X = X[permutation]
        z = z[permutation]
        # Iterate over the points
        for i in range(num_pts):
            # Fill in here
            x_i = X[i,:] # all vars for a given point
            z_i = z[i] #prediction for ith z
            z_hat_i = logistic(np.dot(x_i,params))
            nu = 1/ep
            gradient = 2*(z_hat_i-z_i)*z_hat_i*(1-z_hat_i)*x_i
            params = params - nu*gradient
    return params

def logistic(z):
    s = 1 / (1+np.exp(-z))
    return s