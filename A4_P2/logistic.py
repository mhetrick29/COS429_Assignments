import numpy as np

def logistic(x):
    """The logistic "sigmoid" function"""
    s = 1 / (1+np.exp(-x))
    return s

