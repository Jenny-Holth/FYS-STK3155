"""
Code is made using Python version 3.9.20. 
We have used Python libraries numpy, scikit-learn, matplotlib, seaborn, autograd and numba. 
Code built on other work is cited above relevant classes/functions.
"""
import numpy as npy
from autograd import numpy as anp
from copy import deepcopy
import autograd.numpy as np
from autograd import grad
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPRegressor,  MLPClassifier
from sklearn.linear_model import LogisticRegression
import random
from numba import njit
np.random.seed(2024)
random.seed(2024)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
from Utilities import sigmoid, sigmoid_der, cross_entropy, cross_entropy_der,  ReLU, ReLU_der,leakyReLU,leakyReLU_der,linear,linear_der, R2, mse,mse_der, cost_func, cross_entropy, cross_entropy_der




"""
The codes in the NeuralNetwork class are built on code provided in the exercises of week 42 in FYS-STK3155 at:
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek42.html
"""
class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes,
        activation_funcs, activation_ders, cost_fun, cost_der):
        """
        Initializes the NeuralNetwork class with the given parameters for setting up the neural network architecture.

        Inputs:
        -network_input_size (int): The size of the input layer, representing the number of features in the input data.
        -layer_output_sizes (list/array): List specifying the number of neurons in each layer.
        -activation_funcs (list): List of activation functions for each layer in the network.
        -activation_ders (list): List of derivative functions for each activation function, used for backpropagation.
        -cost_fun (function): Cost function to evaluate model performance.
        -cost_der (function): Derivative of the cost function, used for computing gradients.
        """
                
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
    
    def create_layers_batch(self, SqrtDivide = True):
        """
        Creates and initializes the network layers with weights and biases.
        Inputs:
        -None (uses network initialized in __init__).

        Outputs:
        -layers (list): A list of tuples, where each tuple contains:
        -W (array): Weight matrix for a layer, initialized with a random normal distribution.
        -b (array): Bias vector for the layer, initialized randomly.
        """
        np.random.seed(2024)
        random.seed(2024)
        layers = []
        layer_output_sizes = self.layer_output_sizes
        i_size = self.network_input_size
        if SqrtDivide:
            for layer_output_size in layer_output_sizes:
                W = np.random.randn(layer_output_size, i_size).T  /np.sqrt(layer_output_size)
                b = np.random.randn(layer_output_size)           
                layers.append((W, b))

                i_size = layer_output_size
            return layers
        else:
            for layer_output_size in layer_output_sizes:
                W = np.random.randn(layer_output_size, i_size).T  
                b = np.random.randn(layer_output_size)           
                layers.append((W, b))

                i_size = layer_output_size
            return layers
    
    def feed_forward_batch(self, input, layers): 
        """
        Performs a feed-forward pass through the network with the provided input and layers.

        Inputs:
        -input (array): The input data to be passed through the network.
        -layers (list): The list of layer tuples (weights W and biases b) created in create_layers_batch.

        Outputs:
        -a (array): The output of the last layer (network output).
        """
        a = input  
        for idx, ((W, b), activation_func) in enumerate(zip(layers, self.activation_funcs)):
            z = a @ W + b 
            a = activation_func(z)  

        return a    
    
    def feed_forward_saver_batch(self, input, layers):
        """
        Performs a feed-forward pass through the network, saving intermediate values for backpropagation.

        Inputs:
        -input (array): The input data for the network.
        -layers (list): The list of layer tuples (weights W and biases b).

        Outputs:
        -layer_inputs (list): List of intermediate activations for each layer.
        -zs (list): List of pre-activation values (z) for each layer.
        -a (array): The output of the final layer (network prediction).
        """
        activation_funcs = self.activation_funcs
        layer_inputs = []
        zs = []
        a = input
        for (W, b), activation_func in zip(layers, activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            #layer_inputs.append(a)
            zs.append(z)

        return layer_inputs, zs, a

    def backpropagation_batch(self, input, target, layers):
        """
        Performs backpropagation to calculate gradients for each layer based on a batch of input data and targets.

        Inputs:
        -input (array): The input data for training.
        -target (array): The target or expected output values for the given input data.
        -layers (list): List of layer tuples (weights W and biases b).

        Outputs:
        -layer_grads (list): A list of tuples, where each tuple contains:
        -dC_dW (array): Gradient of the cost with respect to weights.
        -dC_db (array): Gradient of the cost with respect to biases.
        """

        layer_inputs, zs, a = self.feed_forward_saver_batch(input, layers)
        layer_grads = [() for layer in layers]
        cost_der = self.cost_der
        activation_funcs = self.activation_funcs
        activation_ders = self.activation_ders

        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

            if i == len(layers) - 1:  
                dC_da = cost_der(activation_funcs[i](z), target)

            else:
                (W, b) = layers[i + 1]

                dC_da = dC_dz @ W.T

            dC_dz = dC_da * activation_der(z)
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.mean(dC_dz, axis=0)  
            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    def train_network(self, inputs, targets, learning_rate, epochs):
        """
        Trains the neural network on the provided data by performing multiple epochs of feed-forward and backpropagation.

        Inputs:
        -inputs (array): Training data inputs.
        -targets (array): Training data target labels.
        -learning_rate (float): The learning rate for weight updates.
        -epochs (int): The number of training epochs.

        Outputs:
        -layers (list): The trained layers, each containing updated weights W and biases b.
        """

        layers = self.create_layers_batch()
        for _ in range(epochs):
            layers_grad = self.backpropagation_batch(inputs, targets, layers)
            for (W, b), (W_g, b_g) in zip(layers, layers_grad):
                W -= learning_rate*W_g
                b -= learning_rate*b_g
        return layers


"""
The codes in the gradient descent class are built on code examples provided by Morten Hjorth-Jensen used in exercises of week 41 in FYS-STK3155 at: 
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek41.html
"""
class Gradient(NeuralNetwork):
    def __init__(self, X, Target, initial_conditions):
        """
        Initializes the Gradient class as a subclass of NeuralNetwork for optimization with gradient-based methods.

        Inputs:
        -X (array): Input data for training.
        -Target (array): Target values for training.
        """
        np.random.seed(2024)
        random.seed(2024)
        super().__init__(
            network_input_size=X.shape[1],
            layer_output_sizes=initial_conditions.get("layer_output_sizes"),
            activation_funcs=initial_conditions.get("activation_funcs", [sigmoid, sigmoid, sigmoid]),
            activation_ders=initial_conditions.get("activation_ders", [sigmoid_der, sigmoid_der, sigmoid_der]),
            cost_fun=initial_conditions.get("cost_fun", cross_entropy),
            cost_der=initial_conditions.get("cost_der", cross_entropy_der)
        )
        self.cost_fun=initial_conditions.get("cost_fun", cross_entropy)
        self.cost_der=initial_conditions.get("cost_der", cross_entropy_der)
        self.eta_normal = initial_conditions.get("eta_Normal")
        self.eta_stochastic = initial_conditions.get("eta_Stochastic")
        self.initial_conditions = initial_conditions
        self.X = X; self.Target = Target
        self.n, self.p = X.shape
        self.Delta = initial_conditions.get("Delta", 1e-7)
        self.rho = initial_conditions.get("rho", 0.9)
        self.max_iter = initial_conditions.get("max_iter", 3000)
        self.beta1 = initial_conditions.get("beta1", 0.9)
        self.beta2 = initial_conditions.get("beta2", 0.999)
        self.batch_size = initial_conditions.get("batch_size", 10)
        self.tolerance = initial_conditions.get("tolerance", 1e-4)
        self.use_momentum = initial_conditions.get("use_momentum", True)
        self.lamda = initial_conditions.get("lamda", 0.0)
        self.dm = initial_conditions.get("dm", 0.1)

    def set_params(self, **kwargs):
        """
        Updates optimizer parameters in the Gradient instance based on provided keyword arguments.
        
        Inputs:
        -kwargs (dict): Key-value pairs of parameters (for example max_iter=1000, tolerance=1e-5) to override defaults.
        
        Outputs:
        - Updates instance attributes and initial_conditions with new parameter values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)  # Update attribute
                self.initial_conditions[key] = value

    def initialize_optimizer_parameters(self, layers, gradient_type):
        """
        Initializes parameters for gradient-based optimization, based on the gradient type (Normal, AutoGrad, Back).

        Inputs:
        -layers (list): List of network layers, including weights and biases.
        -gradient_type (str): Type of gradient calculation ("Normal", "AutoGrad", "Back").
        
        Outputs:
        -Sets up initial conditions for optimizer parameters (like momentum, squared gradients).
        """
        np.random.seed(2024)
        random.seed(2024)
        if gradient_type == "Normal" or "AutoGrad":
            self.theta = np.random.rand(self.p).reshape(-1,1)
            self.thetaprevious = self.theta.copy()
            self.change = np.zeros_like(self.theta)  
            self.grad_squared = np.zeros_like(self.theta)
            self.first_moment = np.zeros_like(self.theta)
            self.second_moment = np.zeros_like(self.theta)

        if gradient_type == "Back":
            self.theta = np.random.rand(self.network_input_size)
            self.thetaprevious = self.theta.copy()
            self.grad_squared_W = [np.zeros_like(W) for W, b in layers]
            self.grad_squared_b = [np.zeros_like(b) for W, b in layers]
            self.first_moment_W = [np.zeros_like(W) for W, b in layers]
            self.second_moment_W = [np.zeros_like(W) for W, b in layers]
            self.first_moment_b = [np.zeros_like(b) for W, b in layers]
            self.second_moment_b = [np.zeros_like(b) for W, b in layers]
            self.change_W = [np.zeros_like(W) for W, b in layers]  
            self.change_b = [np.zeros_like(b) for W, b in layers] 

    def get_eta(self, command, method):
        """
        Returns the learning rate (eta) based on the command and method.

        Inputs:
        -command (str): Optimization command ("GD","AdaGrad", "Adam", "RMSprop").
        -method (str): Type of optimization ("Normal" or "Stochastic").
        
        Outputs:
        -eta (float): Learning rate for the specified command and method.
        """
        if method == "Normal":
            eta = self.eta_normal.get(command, 0.01)
            return eta
        if method == "Stochastic":
            eta = self.eta_stochastic.get(command, 0.01)
            return eta
        
    def check_tolerance(self, thetaprevious, theta, tolerance):
        """
        Checks if the current parameters meet the tolerance condition for convergence.

        Inputs:
        -thetaprevious (array or list): Parameters from the previous iteration.
        -theta (array or list): Current parameters.
        -tolerance (float): Tolerance for acceptable parameter change.
        
        Outputs:
        -True if the change is within tolerance, otherwise False.
        """

        if isinstance(thetaprevious, np.ndarray) and isinstance(theta, np.ndarray):
            return np.max(np.abs(thetaprevious - theta)) <= tolerance

        elif isinstance(thetaprevious, list) and isinstance(theta, list) and len(thetaprevious) == len(theta):

            flat_curr = np.mean(np.concatenate([w.ravel() for W_b in theta for w in W_b]))
            flat_prev = np.mean(np.concatenate([w.ravel() for W_b in thetaprevious for w in W_b]))
            if np.max(np.abs(flat_prev - flat_curr)) <= tolerance:
                return True
        return False

    def Optimize_Normal(self, Eta=None, lamda=0, command="GD", gradient_type = "Normal"):
        """
        Runs normal (batch) optimization based on the specified command.

        Inputs:
        -lamda (float): Regularization parameter.
        -Eta (float): Learning rate.
        -command (str): Optimization command ("GD", "Adam", etc).
        -gradient_type (str): Type of gradient calculation ("Normal", "AutoGrad").

        Outputs:
        -thetagd (array): Optimized parameters.
        """

        if isinstance(Eta, dict):
            Eta = self.get_eta(command, "Normal")
        else:
            Eta = Eta

        X = self.X; Target = self.Target; self.iter = 0; test = 2; dm = self.dm
        if gradient_type=="Normal" or gradient_type == "AutoGrad":
            self.initialize_optimizer_parameters(0, gradient_type)
            thetagd = self.theta.copy()
        else:
            thetagd = super().create_layers_batch()
            self.initialize_optimizer_parameters(thetagd, gradient_type)

        thetaprevious = np.zeros_like(thetagd for layers in thetagd)
        while np.max(test) >= self.tolerance:
            self.iter += 1

            gradients = self.gradient_types(gradient_type, X, Target, thetagd, lamda, self.n)

            if command == "GD":

                thetagd = self.gd(thetagd, gradients, Eta, dm, gradient_type)
            elif command == "AdaGrad":
                thetagd = self.adagrad(thetagd, gradients, Eta, dm, gradient_type)
            elif command == "Adam":
                thetagd = self.adam(thetagd, gradients, Eta, dm, gradient_type)
            elif command == "RMSprop":
                thetagd = self.rmsprop(thetagd, gradients, Eta, dm, gradient_type)
            else:
                raise ValueError(f"{command} is not supported")

            if self.iter % 50 == 0:
                if self.check_tolerance(thetaprevious, thetagd, self.tolerance) or self.iter>self.max_iter:
                    break
                thetaprevious = deepcopy(thetagd)
            
        return thetagd

    def Optimize_Stochastic(self, lamda=0, Eta=None, command="GD", gradient_type = "Normal"):
        """
        Runs stochastic optimization using mini-batches.

        Inputs:
        -lamda (float): Regularization parameter.
        -Eta (float): Learning rate.
        -command (str): Optimization command ("GD", "Adam", etc).
        -gradient_type (str): Type of gradient calculation ("Normal", "AutoGrad").

        Outputs:
        -thetas (array): Optimized parameters.
        """
        if isinstance(Eta, dict):
            Eta = self.get_eta(command,"Stochastic")
        else:
            Eta = Eta

        X = self.X; Target = self.Target; batch_size = self.batch_size; n = self.n
        m = int(n / batch_size); dm = self.dm
        test = 2
        self.iter = 0

        if gradient_type == "Normal" or gradient_type== "AutoGrad":
            self.initialize_optimizer_parameters(0, gradient_type)
            thetas = self.theta.copy()
        else:
            thetas = super().create_layers_batch() 
            self.initialize_optimizer_parameters(thetas, gradient_type)
        
        indices = np.arange(n)
        thetaprevious = deepcopy(thetas)
        while np.max(test) >= self.tolerance:
            self.iter += 1
            np.random.shuffle(indices)
            for i in range(m):          
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                xi = X[batch_indices]
                Targeti = Target[batch_indices]

                gradients = self.gradient_types(gradient_type, xi, Targeti, thetas, lamda, batch_size)

                if command == "GD":
                    thetas = self.gd(thetas, gradients, Eta, dm, gradient_type)
                elif command == "AdaGrad":
                    thetas = self.adagrad(thetas, gradients, Eta, dm, gradient_type)
                elif command == "Adam":
                    thetas = self.adam(thetas, gradients, Eta, dm, gradient_type)
                elif command == "RMSprop":
                    thetas = self.rmsprop(thetas, gradients, Eta, dm, gradient_type)
                else:
                    raise ValueError(f"{command} is not supported")

                if self.iter % 50 == 0:
                    if self.check_tolerance(thetaprevious, thetas, self.tolerance) or self.iter > self.max_iter:
                        test = [0]
                        break
                thetaprevious = deepcopy(thetas) 
        
        return thetas
    
    def gradient_types(self, type, X, Target, theta, lamda, size):
        """
        Selects the appropriate gradient type (Normal, Backpropagation, AutoGrad).

        Inputs:
        -type (str): Type of gradient ("Normal", "Back", "AutoGrad").
        -X (array): Input data.
        -Target (array): Target values.
        -theta (array): Parameters.
        -lamda (float): Regularization parameter.
        -size (int): Size of the data.

        Outputs:
        -gradients (array or list): Calculated gradients based on selected type.
        """

        if type == "Normal":
            gradients = (2.0 / size) * X.T @ (X @ theta - Target) + 2 * lamda * theta
            return gradients 
        
        elif type == "Back":
            gradient_layers = super().backpropagation_batch(X, Target, theta)
            return gradient_layers

        elif type == "AutoGrad":
            grad_cost = grad(self.cost_fun, argnum=2)
            gradients = 1/size * grad_cost(X, Target, theta, lamda)
            return gradients

    def gd(self, theta, gradients, eta, dm, gradient_type):
        """
        Applies gradient descent update rule to parameters.

        Inputs:
        -theta (array): Current parameters.
        -gradients (array): Gradients.
        -eta (float): Learning rate.
        -dm (float): Momentum decay rate.
        -gradient_type (str): Type of gradient.

        Outputs:
        -theta (array): Updated parameters.
        """
        if gradient_type == "Normal" or gradient_type =="AutoGrad":
            if self.use_momentum:
                _, new_change = fast_update(theta, gradients, 1, eta, 0, self.change, dm)
                theta -= new_change
                self.change = new_change
            else:
                theta -= eta * gradients

            return theta

        elif gradient_type == "Back":
            for j, ((W, b), (W_g, b_g)) in enumerate(zip(theta, gradients)):
                if self.use_momentum:
                    _, W_change = fast_update(W, W_g, 1, eta, 0, self.change_W[j], dm)
                    _, b_change = fast_update(b, b_g, 1, eta, 0, self.change_b[j], dm)
                    W -= W_change
                    b -= b_change
                    self.change_W[j] = W_change
                    self.change_b[j] = b_change
                else:

                    W -= eta * W_g
                    b -= eta * b_g
        return theta

    def adagrad(self, theta, gradients, eta, dm, gradient_type):
        """
        Applies the AdaGrad update rule to parameters.

        Inputs:
        -theta (array): Current parameters.
        -gradients (array): Gradients.
        -eta (float): Learning rate.
        -dm (float): Momentum decay rate.
        -gradient_type (str): Type of gradient.

        Outputs:
        -theta (array): Updated parameters.
        """

        if gradient_type == "Normal" or gradient_type =="AutoGrad":

            self.grad_squared += gradients**2
            if self.use_momentum:
                _, new_change = fast_update(theta, gradients, self.grad_squared, eta, self.Delta, self.change, dm)
                theta -= new_change
                self.change = new_change
            else:

                theta -= eta * gradients / (self.Delta + np.sqrt(self.grad_squared))
            return theta

        elif gradient_type == "Back":
            for j, ((W, b), (W_g, b_g)) in enumerate(zip(theta, gradients)):
                self.grad_squared_W[j] += W_g**2
                self.grad_squared_b[j] += b_g**2

                if self.use_momentum:
                    _, W_change = fast_update(W, W_g, self.grad_squared_W[j], eta, self.Delta, self.change_W[j], dm)
                    _, b_change = fast_update(b, b_g, self.grad_squared_b[j], eta, self.Delta, self.change_b[j], dm)                    
                    
                    W -= W_change
                    b -= b_change
                    self.change_W[j] = W_change
                    self.change_b[j] = b_change

                else:
                    W -= eta * W_g/ (self.Delta + np.sqrt(self.grad_squared_W[j]))
                    b -= eta * b_g/ (self.Delta + np.sqrt(self.grad_squared_b[j]))

            return theta

    def adam(self, theta, gradients, eta, dm, gradient_type):
        """
        Applies the Adam update rule to parameters.

        Inputs:
        -theta (array): Current parameters.
        -gradients (array): Gradients.
        -eta (float): Learning rate.
        -dm (float): Momentum decay rate.
        -gradient_type (str): Type of gradient.

        Outputs:
        -theta (array): Updated parameters.
        """
        if gradient_type == "Normal" or gradient_type =="AutoGrad":
            self.first_moment = self.beta1 * self.first_moment + (1 - self.beta1) * gradients
            self.second_moment = self.beta2 * self.second_moment + (1 - self.beta2) * (gradients ** 2)
            first_unbiased = self.first_moment / (1 - self.beta1 ** self.iter)
            second_unbiased = self.second_moment / (1 - self.beta2 ** self.iter)
            
            if self.use_momentum:
                _ , new_change = fast_update(theta, first_unbiased, second_unbiased, eta, self.Delta, self.change, dm)
                theta -= new_change
                self.change = new_change
            else:
                theta -= eta * first_unbiased / (np.sqrt(second_unbiased) + self.Delta)
            
            return theta
        
        elif gradient_type == "Back":

            for j, ((W, b), (W_g, b_g)) in enumerate(zip(theta, gradients)):

                self.first_moment_W[j] = self.beta1 * self.first_moment_W[j] + (1 - self.beta1) * W_g
                self.second_moment_W[j] = self.beta2 * self.second_moment_W[j] + (1 - self.beta2) * (W_g ** 2)

                self.first_moment_b[j] = self.beta1 * self.first_moment_b[j] + (1 - self.beta1) * b_g
                self.second_moment_b[j] = self.beta2 * self.second_moment_b[j] + (1 - self.beta2) * (b_g ** 2)                

                first_unbiased_W = self.first_moment_W[j] / (1 - self.beta1 ** (self.iter + 1))
                second_unbiased_W = self.second_moment_W[j] / (1 - self.beta2 ** (self.iter + 1))

                first_unbiased_b = self.first_moment_b[j] / (1 - self.beta1 ** (self.iter + 1))
                second_unbiased_b = self.second_moment_b[j] / (1 - self.beta2 ** (self.iter + 1))



                if self.use_momentum:
                    _, W_change = fast_update(W, first_unbiased_W, second_unbiased_W, eta, self.Delta, self.change_W[j], dm)
                    _, b_change = fast_update(b, first_unbiased_b, second_unbiased_b, eta, self.Delta, self.change_b[j], dm)

                    W -= W_change
                    b -= b_change
                    self.change_W[j] = W_change
                    self.change_b[j] = b_change
                
                else:
                    W -= eta * first_unbiased_W / (np.sqrt(second_unbiased_W) + self.Delta)
                    b -= eta * first_unbiased_b / (np.sqrt(second_unbiased_b) + self.Delta)
 
            return theta

    def rmsprop(self, theta, gradients, eta, dm, gradient_type):
        """
        Applies the RMSprop update rule to parameters.

        Inputs:
        -theta (array): Current parameters.
        -gradients (array): Gradients.
        -eta (float): Learning rate.
        -dm (float): Momentum decay rate.
        -gradient_type (str): Type of gradient.

        Outputs:
        -theta (array): Updated parameters.
        """
        rho = self.rho

        if gradient_type == "Normal" or gradient_type =="AutoGrad":

            self.grad_squared = rho * self.grad_squared + (1 - rho) * (gradients**2)
            
            if self.use_momentum:
                _ , new_change = fast_update(theta, gradients, self.grad_squared, eta, self.Delta, self.change, dm)
                theta -= new_change
                self.change = new_change
            else:
                theta -= eta * gradients / (np.sqrt(self.grad_squared) + self.Delta)
            
            return theta


        elif gradient_type == "Back":
            for j, ((W, b), (W_g, b_g)) in enumerate(zip(theta, gradients)):

                g_s_W = rho * self.grad_squared_W[j] + (1 - rho) * (W_g**2)
                g_s_b = rho * self.grad_squared_b[j] + (1 - rho) * (b_g**2)
                self.grad_squared_W[j] = g_s_W
                self.grad_squared_b[j] = g_s_b

                if self.use_momentum:
                    _, W_change = fast_update(W, W_g, self.grad_squared_W[j], eta, self.Delta, self.change_W[j], dm)
                    _, b_change = fast_update(b, b_g, self.grad_squared_b[j], eta, self.Delta, self.change_b[j], dm)
                    W -= W_change
                    b -= b_change
                    self.change_W[j] = W_change
                    self.change_b[j] = b_change

                else:
                    W -= (eta * W_g) / (self.Delta + np.sqrt(g_s_W))
                    b -= (eta * b_g) / (self.Delta + np.sqrt(g_s_b))

            return theta

    def mse_lambda_eta(self, lamda_list, eta_list, nor_stok,Gradient_type, command, X_test, Targets_test):
        """
        Computes the mean squared error matrix for different lambda and eta values.

        Inputs:
        -lamda_list (list): List of lambda values.
        -eta_list (list): List of eta values.
        -nor_stok (str): Optimization method ("Normal" or "Stochastic").
        -Gradient_type (str): Type of gradient.
        -command (str): Optimization command.
        -dm (float): Momentum decay rate.

        Outputs:
        -mse_matrix (array): Mean squared error values for each lambda and eta combination.
        -result_list (array): List of optimized parameters for each combination.
        """

        result_list = np.zeros((len(lamda_list), len(eta_list), self.p))
        mse_matrix = np.zeros((len(lamda_list), len(eta_list)))
        
        optimize_method = {
            "Normal": self.Optimize_Normal,
            "Stochastic": self.Optimize_Stochastic
        }
        chosen_optimize = optimize_method[nor_stok]
        
        for n, eta in enumerate(eta_list):
            for k, lamda in enumerate(lamda_list):
                self.lamda = lamda
                self.eta = eta
                theta = chosen_optimize(lamda=lamda, Eta=eta, gradient_type=Gradient_type, command=command)
                Targetpred = X_test @ theta
                Mse = mse(Targetpred, Targets_test)
                mse_matrix[n, k] = Mse

        return mse_matrix, result_list

    #Find best hyperparameters
    def find_hyperparameters(self, matrix, eta_list, lamda_list):
        min_mse = np.min(matrix)
        min_index = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
        best_lambda = lamda_list[min_index[0]]
        best_eta = eta_list[min_index[1]]
        return best_lambda, best_eta, min_mse


@njit
def fast_update(W, grad, grad_squared, eta, Delta, momentum, dm):
    new_change = eta * grad / (Delta + npy.sqrt(grad_squared)) + dm * momentum
    W = 0
    return W, new_change



def initialize_classes(initial_conditions, X, Target, custom_conditions=None):
    """
    Initializes the `NeuralNetwork` and `Gradient` classes based on default and custom configurations.
    
    Inputs:
    -initial_conditions (dict): Default conditions for network and optimizer configurations.
    -X (array): Input data for training.
    -Target (array): Target values for training.
    -custom_conditions (dict, optional): Conditions to override default initial_conditions.

    Outputs:
    -neural_net (NeuralNetwork): Initialized neural network.
    -optimizer (Gradient): Initialized optimizer for the neural network.
    """
    
    #Merge initial conditions with any custom conditions provided
    conditions = initial_conditions.copy()
    if custom_conditions:
        conditions.update(custom_conditions)
    initial_conditions = conditions
    #Extract parameters from merged conditions
    layer_output_sizes = conditions.get("layer_output_sizes")
    activation_funcs = conditions.get("activation_funcs", [sigmoid, sigmoid, sigmoid])
    activation_ders = conditions.get("activation_ders", [sigmoid_der, sigmoid_der, sigmoid_der])

    #Initialize the neural network with specified parameters
    neural_net = NeuralNetwork(
        network_input_size = X.shape[1],
        layer_output_sizes = layer_output_sizes,
        activation_funcs = activation_funcs,
        activation_ders = activation_ders,
        cost_fun = conditions.get("cost_fun", cross_entropy),
        cost_der = conditions.get("cost_der", cross_entropy_der),
    )

    optimizer = Gradient(
            X=X,
            Target=Target,
            initial_conditions=conditions,
          )

    #Set additional parameters in the optimizer
    optimizer.set_params(
        batch_size=conditions.get("batch_size"),
        B=conditions.get("B"),
        tolerance=conditions.get("tolerance"),
        beta1=conditions.get("beta1"),
        max_iter=conditions.get("max_iter"),
        beta2=conditions.get("beta2"),
        use_momentum=conditions.get("use_momentum"),
        rho=conditions.get("rho")
    )

    return neural_net, optimizer
