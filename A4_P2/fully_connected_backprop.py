def fully_connected_backprop(dLdU, x, W):
    import numpy as np
    # Computes the derivates of the loss with respect to the input x,
    # weights W, and biases b.
    #
    # dLdU: a row-vector with shape [1, neuron_count] that contains the
    #   derivative of the loss with respect to each element of the output of
    #   the fully connected layer.
    # x: a row-vector with shape [1, feature_count]. The input example that was
    #   passed into the layer. This is either the original training feature
    #   vector, if this is the first layer of the network, or the activations
    #   of the previous layer of the network if this layer is further in.
    # W: a matrix with shape [feature_count, neuron_count]. The current weight
    #   matrix at this layer.
    # [not provided: b]. A row vector with shape [1, neuron_count]. The current
    #   bias vector at this layer. Hint: We don't provide it because the
    #   derivatives work out such that it isn't necessary.
    # return:
    # [dLdX]: a row-vector with shape [1, feature_count] (the same shape as
    #   x). Each element of this vector should contain the partial derivative
    #   of L with respect to the corresponding value of the input x.
    # [dLdW]: a matrix with shape [feature_count, neuron_count] (the same shape
    #   as W). Each element should contain the partial derivative of L with
    #   respect to the corresponding value of W. Hint: While a matrix shape on
    #   output is necessary to update W, much of the matrix calculus can be
    #   simplified if you work with a row vector of shape
    #   [1,feature_count*neuron_count] internally.
    # [dLdB]: a row-vector with shape [1, neuron_count] (the same shape as b).
    #   Each element should contain the partial derivative of L with respect to
    #   the corresponding value of b.
    # TODO: Implement me!

    # From Office Hours, dLdX = dLdU*dUdX by chain rule for u = W*x + b
    # The other two gradients follow as well
    dLdX = dLdU @ np.transpose(W)
    dLdW = np.transpose(x) @ dLdU
    dLdB = dLdU
    return dLdX, dLdW, dLdB
