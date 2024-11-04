"""
Code is made using Python version 3.9.20. 
"""
import numpy as np
import autograd.numpy as np
"""
Common utility functions for activation functions, cost functions, and performance metrics.

Inputs:
-z (array): Input to activation functions or their derivatives.
-predict (array): Predicted values from the model.
-target (array): Actual target values.
-X (array): Input data.
-theta (array): Parameter weights.
-lamda (float): Parameter for ridge cost functions.

Outputs:
- Activation Functions:
    -leakyReLU(z): Computes the Leaky ReLU activation on z.
    -leakyReLU_der(z): Derivative of Leaky ReLU for backpropagation.
    -ReLU(z): Computes the ReLU activation on z.
    -ReLU_der(z): Derivative of ReLU for backpropagation.
    -sigmoid(z): Computes the sigmoid activation on z, with overflow handling.
    -sigmoid_der(z): Derivative of sigmoid for backpropagation.
    -linear(z): Linear activation function on z.
    -linear_der(z): Derivative of linear function (constant).

- Performance Metrics:
    -R2(predict, target): Calculates R-squared for predict and target.
    -mse(predict, target): Mean squared error for predict and target.
    -mse_der(predict, target): Derivative of mean squared error for backpropagation.

- Cost Functions:
    -cross_entropy(predict, target): Calculates the cross-entropy cost for binary classification.
    -cross_entropy_der(y_pred, y_true): Derivative of cross-entropy cost for backpropagation.
    -cost_func(X, target, theta, lamda): Computes regularized mean squared error cost.
"""

def leakyReLU(z):
  A = 0.2
  return np.where(z > 0, z, A*z)

def leakyReLU_der(z):
  A = 0.2
  return np.where(z > 0, 1, A*z)

def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    clipping = 250
    z = np.clip(z, -clipping, clipping) #Clipping z so that we dont encounter overflow
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    clipping = 250
    z = np.clip(z, -clipping, clipping) #Clipping z so that we dont encounter overflow
    return np.exp(-z)/ (1 + np.exp(-z))**2

def linear(z):
    return z

def linear_der(z):
    return 1

def R2(predict, target):
    return 1 - np.sum((target-predict)**2)/(np.sum((target-np.mean(predict))**2))

def mse(predict, target):
    return np.mean((predict - target)**2)

def mse_der(predict, target):
    return  2/predict.size * (predict - target)

def cross_entropy(predict , target):
    return -np.mean(target * np.log(predict) + (1 - target) * np.log(1 - predict))

def cross_entropy_der(y_pred, y_true):
    return - 1/y_pred.size * ((y_true * 1/(y_pred + 1e-10)) - (1-y_true)/(1-y_pred + 1e-10))

def cost_func(X, target, theta, lamda):
    return np.sum((target- X@ theta)**2 ) + lamda * np.sum(theta ** 2)
