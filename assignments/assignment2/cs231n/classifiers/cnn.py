from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.C = input_dim[0]
        self.F = num_filters
        self.K = filter_size
        self.H = input_dim[1]
        self.W = input_dim[2]
        self.pad = (filter_size - 1) // 2
        self.stride = 1
        self.pool_height = 2
        self.pool_width = 2
        self.pool_stride = 2
        self.H_out = 1 + (self.H + 2 * self.pad - self.K) // self.stride
        self.W_out = 1 + (self.W + 2 * self.pad - self.K) // self.stride
        self.H_pool = 1 + (self.H_out - self.pool_height) // self.pool_stride
        self.W_pool = 1 + (self.W_out - self.pool_width) // self.pool_stride
        self.params["W1"] = np.random.normal(0, 1 / (self.K * self.K), (self.F, self.C, self.K, self.K))
        self.params["b1"] = np.zeros(self.F)
        self.params["W2"] = np.random.normal(0, 1 / hidden_dim, (self.F * self.H_pool * self.W_pool, self.hidden_dim))
        self.params["b2"] = np.zeros(self.hidden_dim)
        self.params["W3"] = np.random.normal(0, 1 / num_classes, (self.hidden_dim, self.num_classes))
        self.params["b3"] = np.zeros(self.num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': self.stride, 'pad': self.pad}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': self.pool_height,
                      'pool_width': self.pool_width,
                      'stride': self.pool_stride}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        X, self.cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        X, self.cache2 = affine_relu_forward(X, W2, b2)
        scores, self.cache3 = affine_forward(X, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        dX, dW3, db3 = affine_backward(dscores, self.cache3)
        dW3 += self.reg * W3
        dX, dW2, db2 = affine_relu_backward(dX, self.cache2)
        dW2 += self.reg * W2
        dX, dW1, db1 = conv_relu_pool_backward(dX, self.cache1)
        dW1 += self.reg * W1
        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        grads["W3"] = dW3
        grads["b3"] = db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
