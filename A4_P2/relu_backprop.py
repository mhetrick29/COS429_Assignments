def relu_backprop(dLdy, x):
    import numpy as np
    # Backpropogates the partial derivatives of loss with respect to the output
    # of the relu function to compute the partial derivatives of the loss with
    # respect to the input of the relu function. Note that even though relu is
    # applied elementwise to matrices, relu_backprop is consistent with
    # standard matrix calculus notation by making the inputs and outputs row
    # vectors.
    # dLdy: a row vector of doubles with shape [1, N]. Contains the partial
    #   derivative of the loss with respect to each element of y = relu(x).
    # x: a row vector of doubles with shape [1, N]. Contains the elements of x
    #   that were passed into relu().
    # return:
    # [dLdX]: a row vector of doubles with shape [1, N]. Should contain the
    #   partial derivative of the loss with respect to each element of x.

    # TODO: Implement me!
    dLdX = np.multiply(dLdy, dRelu(x))
    return dLdX

# helper function to get all values in x greater than 0 and turn into 1 
# this is 'Z' function
def dRelu(x):
    for i in range(len(x)):
        if (x[0][i] > 0):
            x[0][i] = 1
        else:
            x[0][i] = 0
    return x
