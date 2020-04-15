def relu(x):
    # x: a 2-D double array with arbitrary shape.
    # return: relu(x) as described in class applied elementwise.
    # TODO: Implement me!
    import numpy as np
    
    # make new array to fill
    c = np.zeros(x.shape)

    # fill array with max(0, x)
    relu = np.maximum(c, x)
    return relu
