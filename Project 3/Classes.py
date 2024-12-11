"""
Code is made using Python version 3.12.7, Keras API version 3.7.0 and TensorFlow version 2.18.0
We have used Python libraries pandas, numpy, tensorflow, keras, matplotlib, seaborn and sklearn. 
Code built on other work is cited above relevant classes/functions.
This code is a modified version of code from Project 2, original code can be found at: 
https://github.com/Jenny-Holth/FYS-STK3155/blob/main/Project%202/code/Classes.py
"""
import numpy as np
import random
np.random.seed(2024)
random.seed(2024)
from Utilities import ReLU, ReLU_der, softmax, cat_loss, cat_loss_der

"""
The codes in the NeuralNetwork class are built on code examples used in exercises week 41 and the code provided in the exercises week 42 
by Morten Hjorth-Jensen in FYS-STK3155 at:
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek41.html
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek42.html
"""
class NeuralNetwork:
    def __init__(self, Initial_Conditions, Input, Target):
        """
        Initializes the NeuralNetwork class with the given parameters for setting up the neural network architecture.

        Inputs:
        -Initial_Conditions (dict): Dictionary containing all starting parameters like network_input_size, max_epochs etc.
        -Input (array): Input data for training.
        -Target (array): Target values for training.
        """
        self.Initial_Conditions = Initial_Conditions
        self.network_input_size = Input.shape[1]
        self.layer_output_sizes = Initial_Conditions["layer_output_sizes"]
        self.activation_funcs = Initial_Conditions["activation_funcs"]
        self.activation_ders = Initial_Conditions["activation_ders"]
        self.cost_fun = Initial_Conditions["cost_fun"]
        self.cost_der = Initial_Conditions["cost_der"]

        self.Input = Input; self.Target = Target
        self.n, self.p = Input.shape
        self.Delta = Initial_Conditions["Delta"]
        self.max_epochs = Initial_Conditions["epochs"]
        self.beta1 = Initial_Conditions["beta1"]
        self.beta2 = Initial_Conditions["beta2"]
        self.batch_size = Initial_Conditions["batch_size"]

    def create_layers_batch(self, SqrtDivide = False):
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
        input_size = self.network_input_size

        for output_size in self.layer_output_sizes:
            factor = np.sqrt(output_size) if SqrtDivide else 1
            W = np.random.randn(output_size, input_size).T / factor
            b = np.random.randn(output_size)
            layers.append((W, b))
            input_size = output_size

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
            zs.append(z)

        return layer_inputs, zs, a

    def Optimize_Stochastic(self, Eta=None):
        """
        Runs stochastic optimization using mini-batches.

        Inputs:
        -Eta (float): Learning rate.

        Outputs:
        -thetas (array): Optimized parameters.
        """

        Input = self.Input; Target = self.Target; batch_size = self.batch_size; n = self.n
        m = int(n / batch_size); self.epochs = 0; indices = np.arange(n)

        thetas = self.create_layers_batch() 

        self.first_moment_W = [np.zeros_like(W) for W, b in thetas]
        self.second_moment_W = [np.zeros_like(W) for W, b in thetas]
        self.first_moment_b = [np.zeros_like(b) for W, b in thetas]
        self.second_moment_b = [np.zeros_like(b) for W, b in thetas]
        

        while self.max_epochs >= self.epochs:
            self.epochs += 1
            np.random.shuffle(indices)
            for i in range(m):          
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                Input_i = Input[batch_indices]
                Target_i = Target[batch_indices]

                #gradients = self.gradient_types(gradient_type, xi, Targeti, thetas, lamda, batch_size)
                gradients = self.backpropagation_batch(Input_i, Target_i, thetas)
                thetas = self.adam(thetas, gradients, Eta)
        
        return thetas

    def adam(self, theta, gradients, eta):
        """
        Applies the Adam update rule to parameters.

        Inputs:
        -theta (array): Current parameters.
        -gradients (array): Gradients.
        -eta (float): Learning rate.

        Outputs:
        -theta (array): Updated weights and biases.
        """

        for j, ((W, b), (W_g, b_g)) in enumerate(zip(theta, gradients)):

            self.first_moment_W[j] = self.beta1 * self.first_moment_W[j] + (1 - self.beta1) * W_g
            self.second_moment_W[j] = self.beta2 * self.second_moment_W[j] + (1 - self.beta2) * (W_g ** 2)
            

            self.first_moment_b[j] = self.beta1 * self.first_moment_b[j] + (1 - self.beta1) * b_g
            self.second_moment_b[j] = self.beta2 * self.second_moment_b[j] + (1 - self.beta2) * (b_g ** 2)                

            first_unbiased_W = self.first_moment_W[j] / (1 - self.beta1 ** (self.epochs + 1) + self.Delta)
            second_unbiased_W = self.second_moment_W[j] / (1 - self.beta2 ** (self.epochs + 1) + self.Delta)

            first_unbiased_b = self.first_moment_b[j] / (1 - self.beta1 ** (self.epochs + 1) + self.Delta)
            second_unbiased_b = self.second_moment_b[j] / (1 - self.beta2 ** (self.epochs + 1) + self.Delta)

            W -= eta * first_unbiased_W / (np.sqrt(second_unbiased_W) + self.Delta)
            b -= eta * first_unbiased_b / (np.sqrt(second_unbiased_b) + self.Delta)

        return theta

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
        layer_grads = [() for _ in layers]
        cost_der = self.cost_der
        activation_funcs = self.activation_funcs
        activation_ders = self.activation_ders

        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

            if i == len(layers) - 1:  
                if activation_der is None:
                    dC_dz = activation_funcs[i](z) - target
                else:
                    dC_da = cost_der(activation_funcs[i](z), target)
                    dC_dz = dC_da * activation_der(z)

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

def initialize_classes(Initial_Conditions, Input, Target, Custom_Conditions=None):
    """
    Initializes the `NeuralNetwork` class based on default and custom configurations.
    
    Inputs:
    -Initial_Conditions (dict): Default Conditions for network and optimizer configurations.
    -Input (array): Input data for training.
    -Target (array): Target values for training.
    -Custom_Conditions (dict, optional): Conditions to override default Initial_Conditions.

    Outputs:
    -Neural_Net (NeuralNetwork): Initialized neural network.
    """
    
    #Merge initial Conditions with any custom Conditions provided
    Conditions = Initial_Conditions.copy()
    if Custom_Conditions:
        Conditions.update(Custom_Conditions)
    
    #Initialize the neural network with specified parameters
    Neural_Net = NeuralNetwork(
        Initial_Conditions = Conditions,
        Input = Input,
        Target= Target,
    )

    return Neural_Net
