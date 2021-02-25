"""
Used to run different experiments for the neural network project.
"""

import mnist_loader
import network
import numpy as np

if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    name = "45_epoch"

    net = network.Network([784, 30, 10])
    net.SGD(training_data, 45, 10, 3.0, validation_data)
    acc, cm = net.evaluate(test_data, 1)
    print("{0}: {1}".format(name, acc))
    print(cm)
    net.save_network(name)

    # net = network.load_network("2_hidden_layers")
    # acc, cm = net.evaluate(test_data, 1)
    # print("2 hidden layers: {0}".format(acc))
    # print(cm)

