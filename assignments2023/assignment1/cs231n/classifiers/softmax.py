from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = X.shape
    C = W.shape[1]
    f = np.matmul(X, W) # N x C
    f -= np.max(f, axis=1).reshape((N, 1))
    exp_f = np.exp(f) # N x C
    sigma = np.sum(exp_f, axis=1).reshape((N))
    softmax = exp_f / sigma.reshape((N, 1))
    loss = - np.log(softmax[np.arange(N), y]).sum()
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    softmax[np.arange(N), y] -= 1
    dW = np.matmul(X.reshape(N, D, 1), softmax.reshape(N, 1, C)).sum(axis=0)
    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    f = np.matmul(X, W) # (N, C)
    f -= np.max(f, axis=1, keepdims=True)
    softmax = np.exp(f) / np.exp(f).sum(axis=1, keepdims=True) # (N, C)
    loss = np.sum(-np.log(softmax[np.arange(N), y])) / N + 0.5 * reg * np.sum(W * W)
    softmax[np.arange(N), y] -= 1
    dW = X.T.dot(softmax) / N + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
