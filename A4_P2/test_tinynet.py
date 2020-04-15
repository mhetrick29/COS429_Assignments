#
# Princeton University, COS 429, Fall 2019
#
# test_tinynet.py
#   Test tinynet on some simple data in 'training_pacman.txt'
#   and 'test_pacman.txt'
#
import numpy as np
import matplotlib.pyplot as plt
from tinynet_sgd import tinynet_sgd
from tinynet_predict import tinynet_predict


def test_tinynet():
    # Load the training data
    training = np.loadtxt('training_pacman.txt')

    layers = [32]  # additional layers should be colon separated.

    # Do the training
    example_count = training.shape[0]
    X = training[:, :2]
    z = training[:, 2]
    epoch_count = 5
    net = tinynet_sgd(X, z, layers, epoch_count)

    # Apply the learned model to the training data and print out statistics
    predicted = tinynet_predict(X, net)
    training_accuracy = np.sum(predicted == z) / example_count
    print('training_accuracy =', training_accuracy)

    # Plot "ground truth" and predictions
    plot_classes(training, z, 'Training ground truth', 1)
    plot_classes(training, predicted, 'Training predicted', 2)

    # Apply the learned model to the test data
    testing = np.loadtxt('test_pacman.txt')
    example_count = testing.shape[0]
    X = testing[:, :2]
    z = testing[:, 2]
    predicted = tinynet_predict(X, net)
    testing_accuracy = np.sum(predicted == z) / example_count
    print('testing_accuracy = ', testing_accuracy)

    # Plot "ground truth" and predictions
    plot_classes(testing, z, 'Testing ground truth', 3)
    plot_classes(testing, predicted, 'Testing predicted', 4)


def plot_classes(data, z, name, num):
    """Create a scatterplot of the given data.  It is assumed that the input
       data is 2-dimensional.

    Args:
        data: datapoints (one per row, only first two columns used)
        z: labels (0/1)
        name: name of the figure
        num: number of the figure
    """
    plt.figure(num)
    plt.title(name)
    positive = data[z > 0]
    negative = data[z == 0]
    plt.scatter(positive[:, 0], positive[:, 1], c='red', alpha=0.5)
    plt.scatter(negative[:, 0], negative[:, 1], c='blue', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    test_tinynet()
