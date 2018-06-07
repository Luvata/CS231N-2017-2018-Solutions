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
  num_trains = X.shape[0]
  num_classes = W.shape[1]
  dA = np.zeros((num_trains,num_classes))
  for i in xrange(num_trains):
    dot = np.dot(X[i],W)
    dot -= np.max(dot)
    exp = np.exp(dot)
    norm = exp/ np.sum(exp)
    loss -= np.log(norm[y[i]])
    norm[y[i]] -= 1
    dA[i] = norm
  dW = X.T.dot(dA)
  dW /= num_trains
  dW += 2 * reg * W
  ################## MUST NOT FORGET
  loss/=num_trains #
  ##################
  loss += reg * np.sum(W*W)
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_trains = X.shape[0]
  num_classes = W.shape[1]
  activation_all = np.dot(X,W)
  max_row = np.max(activation_all,axis=1)
  activation_all = (activation_all.T - max_row).T
  exp = np.exp(activation_all)
  norm = (exp.T / np.sum(exp,axis=1)).T # NxC
  loss -= np.sum(np.log(norm[np.arange(num_trains),y]))
  norm[range(num_trains),y] -= 1
  dW = X.T.dot(norm)



  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /= num_trains
  dW += 2 * reg * W
  loss /= num_trains
  loss += reg * np.sum(W*W)
  return loss, dW

