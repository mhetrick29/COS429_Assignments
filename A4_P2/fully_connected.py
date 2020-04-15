# The forward layer for a hidden neuron. 
def fully_connected(x, W, b):
    # x: a row vector with shape [1, feature_count].
    # W: a matrix with shape [feature_count, neuron_count] containing the
    # parameters of the hidden layer.
    # return: a row vector with shape [1, neuron_count] containing the network
    # responses at the hidden layer.
    # TODO: Implement me!

    # The function u = x*W + b
    return (x @ W) + b
