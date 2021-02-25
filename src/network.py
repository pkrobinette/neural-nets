"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

Added Functions
---------------
evaluate_binary
transform_digit 
save_network
roc
load_network
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

from PIL import Image
from PIL import ImageOps

import os
import csv

import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # if test_data:
            #     # print "Epoch {0}: {1} / {2}".format(
            #     #     j, self.evaluate(test_data), n_test)
            #     # print "Epoch {0}:{1}".format(j, self.evaluate(test_data))
            # else:
            #     print "Epoch {0} complete".format(j)
            print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data, cm = 0):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        for i in range(len(test_data)):
            x, y = test_results[i]
            if x != y:
                self.transform_digit(test_data[i], x)
        if cm:
            confuse_matrix = np.zeros((11,11), dtype=int)
        
        # Sets labels in the matrix to make it easier to read
            for x in range(0,10):
                confuse_matrix[0][x+1] = x
                confuse_matrix[x+1][0] = x
            for (x,y) in test_results:
                confuse_matrix[x+1][y+1] +=1 # x is predicted label, and y is actual label

            correct = sum(int(x == y) for (x, y) in test_results)
            acc = float(correct)/len(test_data)
            return acc, confuse_matrix

        return sum(int(x == y) for (x, y) in test_results)
    

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Personal Functions Start Here ------------------------------------------------------------------

    def evaluate_binary(self, test_data, num, thresh):
        """
        Evaluate model on whether it is a number or not based on a given threshold.

        Parameters
        ----------
        test_data : 
            test data from the mnist library extracted with load_data_wrapper()
        
        num : int
            Number to evaluate. In range 0-9.

        thresh : float
            Threshold value used to label a number when compared with feedforward output

        Returns
        -------
        fpr : float
            false positive rate

        tpr : float
            true positive rate
        """
        test_results = [(int(self.feedforward(x)[num]>thresh), y) for (x,y) in test_data]
        total_num_predict = sum(x for x,y in test_results)
        total_num_actual = sum(int(y==num) for x,y in test_results)
        total_not_num_actual = len(test_results) - total_num_actual
        total_num_correct = sum(int(num==y) for (x,y) in test_results if x == 1)

        if total_num_predict == 0:
            perc_pred = 0
        else:
            perc_pred = float(total_num_correct)/total_num_predict

        # true positive rate
        tpr = float(total_num_correct)/total_num_actual
        # false positive rate
        fpr = float(total_num_predict - total_num_correct)/total_not_num_actual

        # print(" Correct/Predicted: {0} / {1} = {2}%".format(total_num_correct, total_num_predict, int(perc_pred*100)))
        # print(" Correct/Actual: {0} / {1} = {2}%".format(total_num_correct, total_num_actual, int(tpr *100)))

        return fpr, tpr

    def transform_digit(self, data, predict):
        """
        Parameters
        ----------
        data : 1x2 list
            A list containing the greyscale value at each pixel in [0]
            element and digit label in [1] element.
        
        predict : int
            What the predicted label of the digit is.
        """
        num = list(data[0])
        num = np.array(num)
        num = num*255
        num = num.astype(np.uint8)
        num = np.reshape(num, (28,28))

        new_image = Image.fromarray(num, mode = "L")
        new_image = ImageOps.invert(new_image)
        new_image.save('./mislabeled_nums/actual_{0}/predict_{1}.png'.format(data[1], predict))
    
    def save_network(self, name):
        """ 
        Saves the size, weights, and biases of the current network
        
        Parameters
        ----------
        name : string
            Name of pre-trained model
        """
        if not os.path.exists('pretrained_networks/{0}'.format(name)):
            os.makedirs('pretrained_networks/{0}'.format(name))

        np.save("pretrained_networks/{0}/sizes.npy".format(name), self.sizes)
        np.save("pretrained_networks/{0}/weights.npy".format(name), self.weights)
        np.save("pretrained_networks/{0}/biases.npy".format(name), self.biases)

    def roc(self, test_data, num):
        """
        Creates an ROC curve from given data for a desired label in the model.

        Parameters
        ----------
        test_data : 
            test data from the mnist library extracted with load_data_wrapper()
        
        num : int
            Number to evaluate. In range 0-9.
        """
        inter = np.linspace(0,1,21)
        roc_vals = [(self.evaluate_binary(test_data, num, thresh)) for thresh in inter]
        fpr, tpr = zip(*roc_vals)
        plt.plot(fpr, tpr, "*-")
        plt.title("ROC Curve for Number: {0}".format(num))
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.savefig("ROC/ROC_{0}.png".format(num))

    

#### Miscellaneous functions
def load_network(name):
        """ 
        Loads a saved network to prevent retraining
        
        Parameters
        ----------
        name : string
            Name of pre-trained model
        
        Return
        ------
        net : Network()
            a nn created from the pretrained model
        """
        net = Network([0])
        net.sizes = np.load("pretrained_networks/{0}/sizes.npy".format(name))
        net.weights = np.load("pretrained_networks/{0}/weights.npy".format(name))
        net.biases = np.load("pretrained_networks/{0}/biases.npy".format(name))
        net.num_layers = len(net.sizes)
        return net

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
