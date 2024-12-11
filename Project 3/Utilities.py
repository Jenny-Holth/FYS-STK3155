"""
Code is made using Python version 3.12.7, Keras API version 3.7.0 and TensorFlow version 2.18.0

Common utility functions for activation functions, cost functions, and performance metrics.

Inputs:
-z (array): Input to activation functions or their derivatives.
-y_pred (array): Predicted values from the model.
-y_true (array): Actual target values.

Outputs:
- Activation Functions:
    -ReLU(z): Computes the ReLU activation on z.
    -ReLU_der(z): Derivative of ReLU for backpropagation.
    -softmax(z): Computes the softmax activation on z.

- Cost Functions:
    -cat_loss(y_pred, y_true): Calculates the cross-entropy cost for multi-class classification.
    -cat_loss_der(y_pred, y_true): Derivative of cat-loss cost for backpropagation.
"""

import numpy as np

def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def softmax(z):
    clipping = 400
    z = np.clip(z, -clipping, clipping)   #Clipping z so that we dont encounter overflow
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def cat_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def cat_loss_der(y_pred, y_true):
    return y_pred - y_true


