"""
Code is made using Python version 3.12.7, Keras API version 3.7.0 and TensorFlow version 2.18.0
We have used Python libraries pandas, numpy, tensorflow, keras, matplotlib, seaborn and sklearn. 
Code built on other work is cited above relevant classes/functions.
"""
import numpy as np 
import tensorflow as tf
import pandas as pd
import timeit
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import datasets, layers, models, optimizers, regularizers
from keras.utils import to_categorical        # type: ignore
from keras.models import Model, Sequential    # type: ignore
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D         # type: ignore
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from Classes import NeuralNetwork, initialize_classes
from Utilities import ReLU, ReLU_der, softmax, cat_loss, cat_loss_der

np.random.seed(2024)
random.seed(2024)
keras.utils.set_random_seed(2024)

def check_accuracy(predictions, targets):
    """
    Checks and calculates the accuracy between predictions and target values, handling both 1D and 2D arrays.

    Inputs:
    -predictions (array): Array of predicted values.
    -targets (array): Array of target values.

    Outputs:
    -accuracy (float): Proportion of correctly predicted values.
    """

    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    if targets.shape[1] > 1:
        targets = np.argmax(targets, axis=1)

    
    accuracy = np.sum(predictions == targets) / len(targets)
    return accuracy

def get_data(DataType, DataAmount):
    """
    Loads and preprocesses the Fashion MNIST dataset based on the specified type and amount of data.

    Inputs:
        DataType (str): Type of preprocessing. Options:
                         - "CNN": Prepares data for convolutional neural networks.
                         - "Skl_LR": Prepares flattened data for scikit-learn Logistic Regression.
                         - Any other string: Prepares flattened data with one-hot encoded targets.
        DataAmount (int): Number of samples to use from the training dataset.

    Outputs:
        tuple: Preprocessed training and test datasets (x_train, x_test, y_train, y_test)
    """

    np.random.seed(2024)
    random.seed(2024)
    keras.utils.set_random_seed(2024)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:DataAmount]
    y_train = y_train[:DataAmount]
    x_test = x_test[:DataAmount//6]
    y_test = y_test[:DataAmount//6]
    n,p,k = x_train.shape

    if DataType == "CNN":
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
        x_train = x_train.reshape(-1, p, k, 1)
        x_test = x_test.reshape(-1, p, k, 1)

    elif DataType == "Skl_LR":
        x_train = x_train.reshape(len(x_train), p*k) / 255.0     # Scaling  
        x_test = x_test.reshape(len(x_test), p*k) / 255.0

    else:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
        x_train = x_train.reshape(len(x_train), p*k) / 255.0             # Scaling
        x_test = x_test.reshape(len(x_test), p*k) / 255.0

    return x_train, x_test, y_train, y_test
    
def save_data(Matrix, Param1, Param1_Values, Param2, Param2_Values, Filename):
    """
    Saves a matrix to a file with labeled rows and columns based on parameter values.

    Imputs:
        Matrix (array): 2D array or matrix to save.
        Param1 (str): Name of the first parameter (for rows).
        Param1_Values (list or array): Values of the first parameter.
        Param2 (str): Name of the second parameter (for columns).
        Param2_Values (list or array): Values of the second parameter.
        Filename (str): Name of the file to save (without extension).
    """
    os.makedirs(".\\data", exist_ok=True)
    df = pd.DataFrame(
        Matrix,
        index=[f"{Param1}={par1}" for par1 in Param1_Values],  
        columns=[f"{Param2}={par2}" for par2 in Param2_Values]  
    )
    df.to_csv(f".\\data\\{Filename}.txt", sep='\t')

def save_load_data_single(Filename, Predictions=None, Targets=None, Scores=None, Runtimes=None, Save=False ):
    if Save:
        df = pd.DataFrame({
            'Prediction': Predictions,
            'Target': Targets,
            'Scores': Scores if isinstance(Scores, list) else [Scores] * len(Predictions),
            'Runtime': Runtimes*(int( (len(Predictions)/2) ) ) if isinstance(Runtimes, list) else [Runtimes] * len(Predictions)
        })
        df.to_csv(f".\\data\\{Filename}.txt", sep='\t')
    if not Save:
        df_loaded = pd.read_csv(f".\\data\\{Filename}.txt", sep='\t')

        
        Predictions = df_loaded['Prediction'].tolist()
        Targets = df_loaded['Target'].tolist()
        Scores = df_loaded['Scores'].tolist()
        Runtimes = df_loaded['Runtime'].tolist()
        return np.array(Predictions), np.array(Targets), np.array(Scores[0]), np.array(Runtimes[:2])

def load_data(Filename):
    """
    Loads a saved matrix file and extracts parameter values from the labels.

    Input:
        filename (str): Name of the file to load.

    Outputs:
        tuple: (matrix, row_labels, column_labels)
            - matrix (array): The loaded matrix as a NumPy array.
            - row_labels (list): Extracted parameter values for rows, converted to int if applicable.
            - column_labels (list): Extracted parameter values for columns, converted to int if applicable.
    """

    df_loaded = pd.read_csv(f".\\data\\{Filename}.txt", sep='\t', index_col=0)
    matrix = df_loaded.values

    # Helper function to check if a value is an integer
    def convert_to_int_or_float(value):
        if value.startswith('('):# and value.endswith(')'):
            return eval(value)
        
        else:
            value_float = float(value)
            if value_float.is_integer():
                return int(value_float)
            return value_float

    # Process row and column labels
    row_labels = np.array([convert_to_int_or_float(label.split('=')[1]) for label in df_loaded.index.tolist()])
    column_labels = np.array([convert_to_int_or_float(label.split('=')[1]) for label in df_loaded.columns.tolist()])
    
    return matrix, row_labels, column_labels

def Create_Keras(input_shape, kernel,n_filters, 
                     n_neurons_connected, n_categories, eta, lmbd, neurons, n_layers, CNN=True):
    """
    Creates and compiles a Keras model for either a Convolutional Neural Network (CNN), Feed Forward Neural Network (FNN) 
    or Logistic Regression (LR).

    Inputs:
    - input_shape (tuple): Shape of the input data.
    - kernel (int or tuple): Size of the convolutional kernel/filter for the CNN layers.
    - n_filters (int): Number of filters for the convolutional layers in the CNN.
    - n_neurons_connected (int): Number of neurons in the fully connected layer for the CNN.
    - n_categories (int): Number of output categories (i.e., number of classes for classification).
    - eta (float): Learning rate for the optimizer.
    - lmbd (float): L2 regularization parameter for the layers.
    - neurons (int): Number of neurons per layer for FCNN when CNN=False.
    - n_layers (int): Number of layers in the FCNN when CNN=False.
    - CNN (bool): If True, creates a CNN. If False, creates a Fully Connected Neural Network.

    Outputs:
    - model (Keras Sequential model): Compiled Keras model ready for training.
    """
    np.random.seed(2024)
    random.seed(2024)
    keras.utils.set_random_seed(2024)
    model = Sequential()

    """
    The following Keras CNN structure follows a code example by Morten Hjorth-Jensen in FYS-STK3155 
    at https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter12.html#the-mnist-dataset-again 
    """
                         
    if CNN:
        model.add(Input(shape=input_shape))
        model.add(Conv2D(n_filters, kernel, padding='same',
                activation='relu', kernel_regularizer=regularizers.l2(lmbd), strides=(1,1) ))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(n_filters*2, kernel, input_shape=input_shape, padding='same',
                activation='relu', kernel_regularizer=regularizers.l2(lmbd), strides=(1,1) ))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten(name='flatten'))  

        model.add(Dense(n_neurons_connected, activation='relu', kernel_regularizer=regularizers.l2(lmbd)))
        model.add(Dense(n_categories, activation='softmax', kernel_regularizer=regularizers.l2(lmbd)))
    
    else:

        Activators = ['relu']*(n_layers-1) + ['softmax']
        Layers = [neurons]*(n_layers-1) + [n_categories]
        for i in range(n_layers):
            model.add(Dense(Layers[i], activation = Activators[i], input_dim = input_shape[0]*input_shape[1] ))
    
    sgd = optimizers.Adam(learning_rate=eta)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def NeuralNet(Initial_Conditions, Custom = None, BothTimes = False):
    """
    Builds and trains a neural network using specified initial and custom conditions.

    Inputs:
    - Initial_Conditions (dict): Dictionary containing default parameters for the network and training process.
        Required keys:
        - "n_layers": Number of layers in the network.
        - "n_categories": Number of output categories.
        - "neurons": Number of neurons in each hidden layer.
        - "data_amount": Number of training samples to use.
        - "eta": Learning rate for the optimizer.

    - Custom (dict, optional): Dictionary containing parameters to override the defaults in 'Initial_Conditions'.
    - BothTimes (bool): If True returns runtime for training and testing seperately

    Outputs:
    - case (array): Network predictions for the test dataset.
    - accuracyNN (float): Accuracy of the model on the test dataset.
    - Y_test (array): Actual target values for the test dataset.
    """
    
    updated_conditions = Initial_Conditions.copy()
    if Custom:
        updated_conditions.update(Custom)

    n_layers = updated_conditions["n_layers"]
    neurons = updated_conditions["neurons"]
    n_categories = updated_conditions["n_categories"]

    if n_layers > 1:
        layer_sizes = [neurons] * (n_layers - 1) + [n_categories]

        activation_funcs = [ReLU] * (n_layers - 1) + [softmax]
        activation_ders = [ReLU_der] * (n_layers - 1) + [None]

        updated_conditions["layer_output_sizes"] = np.array(layer_sizes)
        updated_conditions["activation_funcs"] = activation_funcs
        updated_conditions["activation_ders"] = activation_ders
    else:
        updated_conditions["layer_output_sizes"] = np.array([n_categories])
    

    start = timeit.default_timer()
    X_Train, X_Test, Y_train, Y_test = get_data("NN", updated_conditions["data_amount"])

    Neural_Net = initialize_classes(updated_conditions, X_Train, Y_train, updated_conditions)
    layers = Neural_Net.Optimize_Stochastic(Eta=updated_conditions["eta"])
    
    start_test = timeit.default_timer()
    case = Neural_Net.feed_forward_batch(X_Test, layers)

    stop = timeit.default_timer()

    if BothTimes:
        Runtime = [start_test - start, stop - start_test]
    else: 
        Runtime = stop - start

    accuracyNN = check_accuracy(case, Y_test)
    return case, accuracyNN, Y_test, Runtime

def parameter_matrix(ModelType, param1_name, param1_vals, param2_name, param2_vals, Initial_Conditions, Custom = None, BothTimes = False, PredReturn=False):
    """
    Generates a parameter matrix by evaluating a model's performance and runtime over specified parameter combinations.

    Inputs:
    - ModelType (str): Type of model to evaluate. Options are "CNN", "FNN_LR", or "FNN_LR_Keras".
    - param1_name (str): Name of the first parameter to vary.
    - param1_vals (list): List of values for the first parameter.
    - param2_name (str): Name of the second parameter to vary.
    - param2_vals (list): List of values for the second parameter.
    - Initial_Conditions (dict): Dictionary containing the base model and training conditions.
    - Custom (dict, optional): Dictionary to override default 'Initial_Conditions'.
    - BothTimes (bool): If True returns runtime for training and testing seperately in list, if False return total runtime

    Outputs:
    - results_matrix (array): 2D array of model accuracies for each combination of 'param1_vals' and 'param2_vals'.
    - time_matrix (array): 2D array of runtimes for each combination of 'param1_vals' and 'param2_vals'.
    """
    np.random.seed(2024)
    random.seed(2024)
    keras.utils.set_random_seed(2024)

    results_matrix = np.zeros((len(param1_vals), len(param2_vals)), dtype=float)
    time_matrix = np.zeros((len(param1_vals), len(param2_vals)), dtype=float)
    Pred_Case = []
    True_Case = []
    for i, param1 in enumerate(param1_vals):
        for j, param2 in enumerate(param2_vals):
            start = timeit.default_timer()

            conditions = Initial_Conditions.copy()  
            if Custom:
                conditions.update(Custom)
            conditions[param1_name] = param1
            conditions[param2_name] = param2


            if ModelType == "CNN":
                X_train, X_test, Y_train, Y_test = get_data("CNN", conditions["data_amount"])

                model = Create_Keras(
                    input_shape=conditions["input_shape"],
                    kernel= conditions["kernel"],
                    n_filters=conditions["n_filters"],
                    n_neurons_connected=conditions["n_neurons_connected"],
                    n_categories=conditions["n_categories"],
                    eta=conditions["eta"],
                    lmbd=conditions["l2"],
                    neurons = conditions["neurons"], 
                    n_layers = conditions["n_layers"],
                    CNN=True,
                )

                model.fit(
                    X_train, Y_train,
                    epochs=conditions["epochs"],
                    batch_size=conditions["batch_size"],
                    verbose=0
                )
                scores = model.evaluate(X_test, Y_test, verbose=0)
                results_matrix[i][j] = scores[1]

            elif ModelType == "FNN_LR":

                case, accuracyNN, case_test, _ = NeuralNet(conditions, conditions, BothTimes=BothTimes)
                results_matrix[i][j] = accuracyNN
                
                if PredReturn:
                    Pred_Case.append(case)
                    True_Case.append(case_test)



            elif ModelType == "FNN_LR_Keras":
                X_train, X_test, Y_train, Y_test = get_data("LR", conditions["data_amount"])

                model = Create_Keras(
                    input_shape=conditions["input_shape"],
                    kernel= conditions["kernel"],
                    n_filters=conditions["n_filters"],
                    n_neurons_connected=conditions["n_neurons_connected"],
                    n_categories=conditions["n_categories"],
                    eta=conditions["eta"],
                    lmbd=conditions["l2"],
                    neurons = conditions["neurons"], 
                    n_layers = conditions["n_layers"],
                    CNN=False,
                )

                model.fit(
                    X_train, Y_train,
                    epochs=conditions["epochs"],
                    batch_size=conditions["batch_size"],
                    verbose=0
                )
                scores = model.evaluate(X_test, Y_test, verbose=0)
                
                results_matrix[i][j] = scores[1]

            stop = timeit.default_timer()
            runtime = stop - start
            time_matrix[i][j] = runtime              
    
    if not PredReturn:
        return results_matrix, time_matrix
    else:
        return results_matrix, time_matrix, Pred_Case, True_Case

def find_hyperparameters(Matrix, Param1, Param2, Time_Matrix):
    """
    Finds the best hyperparameters based on the maximum accuracy (or other metric) in the results matrix.

    Inputs:
    - Matrix (2D array): Results matrix containing evaluation metrics.
    - Param1 (list or array): List of values for the first parameter (corresponding to rows in 'Matrix').
    - Param2 (list or array): List of values for the second parameter (corresponding to columns in 'Matrix').
    - Time_Matrix (2D array): Runtime matrix, with the same dimensions as 'Matrix'.

    Outputs:
    - best_Param1 (float): Value of the first parameter corresponding to the best result.
    - best_Param2 (float): Value of the second parameter corresponding to the best result.
    - max_metric (float): Maximum metric value found in the results matrix.
    - runtime (float): Runtime for the best parameter combination.
    """

    min_mse = np.max(Matrix)
    max_index = np.unravel_index(np.argmax(Matrix, axis=None), Matrix.shape)
    best_Param1 = Param1[max_index[0]]
    best_Param2 = Param2[max_index[1]]
    runtime = Time_Matrix[max_index[0], max_index[1]]

    return best_Param1, best_Param2, min_mse, runtime

def automate_parameter_search(ModelType, Save, Filename, Params, Param_Vals, Initial_Conditions, Custom=None, BothTimes=False):
    """
    Automates hyperparameter search, saving/loading results, and visualization.

    Input:
        ModelType (str): The type of model to run ("CNN", etc.).
        Save (bool): If True, runs parameter_matrix and saves results. If False, loads results from file.
        Filename (str): The filename for saving/loading results.
        Params (list of str): Names of 1-2 parameters to vary (e.g., ["eta", "epochs"]).
        Param_Vals (list of arrays): Corresponding values for the parameters (e.g., [eta_vals, epochs_vals]).
        Initial_Conditions (dict): Initial conditions for the model.

    Output:
        result_matrix (np.ndarray): The accuracy matrix.
        param1_vals (array): Values for the first parameter.
        param2_vals (array): Values for the second parameter.
    """
    np.random.seed(2024)
    random.seed(2024)
    keras.utils.set_random_seed(2024)
    if Save:
        #Run parameter_matrix with the provided parameters and save results

    
        result_matrix, time_matrix = parameter_matrix(
            ModelType=ModelType,
            param1_name=Params[0],
            param1_vals=Param_Vals[0],
            param2_name=Params[1],
            param2_vals=Param_Vals[1],
            Initial_Conditions=Initial_Conditions,
            Custom= Custom,
            BothTimes=BothTimes,
        )

        #Save results
        save_data(result_matrix, Params[0], Param_Vals[0], Params[1], Param_Vals[1], Filename)
        Timefile = f"{Filename}_Time"
        save_data(time_matrix, Params[0], Param_Vals[0], Params[1], Param_Vals[1], Timefile)

        return result_matrix, Param_Vals[0], Param_Vals[1], time_matrix


    else:
        #Load results
        result_matrix, param1_vals, param2_vals = load_data(Filename)
        Timefile = f"{Filename}_Time"
        time_matrix, param1_vals, param2_vals = load_data(Timefile)

        return result_matrix, param1_vals, param2_vals, time_matrix

def automate_single_param_search(ModelType, Save, Filename, Param, Param_Values, Initial_Conditions, Custom=None):
    """
    Automates hyperparameter search for a single parameter, saving/loading results, and finding the best parameter value.

    Input:
        ModelType (str): The type of model to run ("CNN", etc.).
        Save (bool): If True, runs parameter_matrix and saves results. If False, loads results from file.
        Filename (str): The filename for saving/loading results.
        Param (str): Name of the parameter to vary (e.g., "eta").
        Param_Values (array): Corresponding values for the parameter (e.g., eta_vals).
        Initial_Conditions (dict): Initial conditions for the model.
        Custom (dict, optional): Custom conditions to override initial conditions.

    Output:
        result_array (list or array): The accuracy results for the parameter values.
        param_values (array): The parameter values tested.
        max_acc (float): The maximum accuracy achieved.
        best_param (float): The parameter value that gave the best accuracy.
    """
    np.random.seed(2024)
    random.seed(2024)
    keras.utils.set_random_seed(2024)

    if Save:
        # Run parameter_matrix with a single parameter
        result_matrix, time_matrix = parameter_matrix(
            ModelType=ModelType,
            param1_name=Param,
            param1_vals=Param_Values,
            param2_name="Something",  # No second parameter
            param2_vals=[0],          # Placeholder value
            Initial_Conditions=Initial_Conditions,
            Custom=Custom
        )


        result_array = result_matrix[:, 0]  
        time_array = time_matrix[:, 0]


        save_data(result_array, Param, Param_Values, "Placeholder", [0], Filename)
        Timefile = f"{Filename}_Time"
        save_data(time_array, Param, Param_Values, "Placeholder", [0], Timefile)

    else:

        result_matrix, Param_Values, _ = load_data(Filename)
        Timefile = f"{Filename}_Time"
        time_matrix, Param_Values, _ = load_data(Timefile)

        result_array = result_matrix[:, 0] 

    #Find max accuracy and corresponding parameter
    max_acc = np.max(result_array)
    best_param = Param_Values[np.argmax(result_array)]

    return result_array, Param_Values, max_acc, best_param

def RunKeras(Initial_Conditions, CNN = False, BothTimes=False):
    """
    Runs a Keras model (CNN or Fully Connected Network) based on the provided initial conditions.

    Inputs:
    - Initial_Conditions (dict): Dictionary containing model parameters and training configurations.
    - CNN (bool): If True, runs a Convolutional Neural Network. If False, runs a Fully Connected Network (Logistic Regression-like).
    - BothTimes (bool): If True returns runtime for training and testing seperately

    Outputs:
    - Y_pred (array): Predictions made by the model on the test dataset.
    - Y_test (array): Actual labels of the test dataset.
    - Scores (list): Evaluation metrics returned by the model.
    - Runtime (float): Time taken to train and evaluate the model, in seconds.
    """
    keras.utils.set_random_seed(2024)
    start = timeit.default_timer()
    DataType = "CNN" if CNN else "LR"

    Model = Create_Keras(input_shape = Initial_Conditions["input_shape"], 
                    kernel = Initial_Conditions["kernel"],
                    n_filters = Initial_Conditions["n_filters"] , 
                    n_neurons_connected = Initial_Conditions["n_neurons_connected"] , 
                    n_categories = Initial_Conditions["n_categories"], 
                    eta = Initial_Conditions["eta"], 
                    lmbd = Initial_Conditions["l2"],
                    neurons = Initial_Conditions["neurons"], 
                    n_layers = Initial_Conditions["n_layers"],
                    CNN=CNN,
                    )

    X_train, X_test, Y_train, Y_test = get_data(DataType, Initial_Conditions["data_amount"])
    Model.fit(X_train, Y_train,
              epochs= Initial_Conditions["epochs"], 
              batch_size = Initial_Conditions["batch_size"], verbose=0)
    
    start_test = timeit.default_timer()
    Y_pred = Model.predict(X_test)
    stop = timeit.default_timer() 
    Scores = Model.evaluate(X_test, Y_test)

    if BothTimes:
        Runtime = [start_test-start, stop-start_test]
    else:
        Runtime = stop-start

    return Y_pred, Y_test, Scores, Runtime

def RewriteDict(Initial, Update):
    """
    Creates a new dictionary by copying the initial dictionary and updating it with values from another dictionary.

    Inputs:
    - Initial (dict): The base dictionary to copy.
    - Update (dict or None): Dictionary with values to update the 'Initial' dictionary

    Outputs:
    - New (dict): A new dictionary that combines 'Initial' and 'Update'.
    """
    New = Initial.copy()
    if Update:
        New.update(Update)
    return New

def FashionPlot():
    """
    Plots sample images from each class in the Fashion-MNIST dataset with labels.
    """
    labels = ['T-shirt/top [0]', 'Trouser [1]', 'Pullover [2]', 
          'Dress [3]', 'Coat [4]', 'Sandal [5]', 'Shirt [6]', 
          'Sneaker [7]', 'Bag [8]', 'Ankle boot [9]']
    
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    label_args = [np.where(Y_train==i)[0] for i in range(10)]

    image_disp_args = [label_args[i][0] for i in range(10)]
    image_disp = X_train[image_disp_args]

    plt.figure(figsize=(10, 10))
    plt.title('Fashion-MNIST classes', fontsize = 25)
    plt.axis('off')

    for i in range(len(image_disp)):
        if i > 7:
            plt.subplot(3, 4, i+2)
        else:
            plt.subplot(3, 4, i+1)
        plt.title(labels[i], size=18)
        plt.axis('off')
        plt.imshow(image_disp[i], cmap = 'grey')
    plt.show()

def Histogram(Accuracies, Runtimes, Labels, Title, FigSize=(10, 6)):
    """
    Creates a bar plot comparing accuracy and runtime for different classification methods.
    
    Args:
        Accuracies (list): A list of accuracy values for each method.
        Runtimes (list): A list of runtime values for each method.
        Labels (list): A list of method names (e.g., ["FFNN Keras", "FFNN Custom"]).
    """
    # Number of methods

    num_methods = len(Accuracies)
    x = np.arange(num_methods)
    bar_width = 0.4
    
    fig, ax1 = plt.subplots(figsize=FigSize)
    ax2 = ax1.twinx()
    
    #Plot accuracy on the left y-axis
    ax1.bar(x - bar_width / 2, Accuracies, bar_width, label='Accuracy',
            edgecolor='black', color='green', alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=18, color='green')
    ax1.set_ylim([0,1])
    ax1.tick_params(axis='y', labelcolor='green',labelsize=15)
    
    #Plot runtime on the right y-axis
    ax2.bar(x + bar_width / 2, Runtimes, bar_width, label='Runtime',
            edgecolor='black', color='blue', alpha=0.7)
    ax2.set_ylabel('Runtime [s]', fontsize=18, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue',labelsize=15)
    
    #Set x-axis labels and ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(Labels, rotation=0, fontsize=15)
    ax1.set_xlabel('Methods', fontsize=18)
    
    #Add title and adjust layout
    plt.title(Title, fontsize=22)

    fig.tight_layout()
    plt.show()

def heatmap_acc(Matrix, Title,Param1Name, Param2Name, Param1, Param2, Figsize = (9,6),  Sub=False):
    """
    Plots heatmaps for accuracy matrices. Supports single or multiple heatmaps with subplots.

    Inputs:
        Matrix (array or list): Accuracy matrix or list of matrices for subplots.
        Title (str or list): Title for the heatmap or list of titles for subplots.
        Param1Name (str or list): Label for the y-axis or list of labels for subplots.
        Param2Name (str or list): Label for the x-axis or list of labels for subplots.
        Param1 (list or list of lists): Y-axis tick labels or list of labels for subplots.
        Param2 (list or list of lists): X-axis tick labels or list of labels for subplots.
        Figsize (tuple): Figure size for the heatmap(s).
        Sub (bool): If True, creates subplots for multiple matrices.
    """

    if not Sub:
        plt.figure(figsize=Figsize)
        ax = sns.heatmap(Matrix,linewidth=.5, annot=True, 
                        fmt = ".3g",annot_kws={"fontsize":13})
        ax.set_yticklabels(labels=Param1, size = 15)
        ax.set_xticklabels(labels=Param2, size = 15)
        ax.collections[0].colorbar.set_label("Accuracy",size=17)
        ax.set_title(f"{Title}",size=22)
        ax.set_ylabel(f"{Param1Name}",size=17)
        ax.set_xlabel(f"{Param2Name}",size=17)
        plt.show()
    
    elif Sub:
        num_heatmaps = len(Matrix)
        if num_heatmaps == 4:
            fig, axes = plt.subplots(2, 2, figsize=Figsize, constrained_layout=True)
            axes = axes.flatten()
        else:

            fig, axes = plt.subplots(1, num_heatmaps, figsize=Figsize, constrained_layout=True)

        for i, ax in enumerate(axes[:num_heatmaps]): 
            sns.heatmap(
                Matrix[i],
                linewidth=0.5,
                annot=True,
                fmt=".3g",
                annot_kws={"fontsize":13},
                ax=ax
            )
            ax.set_yticklabels(labels=Param1[i], size=12)
            ax.set_xticklabels(labels=Param2[i], size=12)
            colorbar = ax.collections[0].colorbar
            colorbar.set_label("Accuracy", size=17)

            vmin, vmax = colorbar.vmin, colorbar.vmax
            ticks = np.linspace(vmin, vmax, num=5)  
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])

            ax.set_title(f"{Title[i]}", size=20)
            ax.set_ylabel(f"{Param1Name[i]}", size=15)
            ax.set_xlabel(f"{Param2Name[i]}", size=15)
        
        # Hide extra axes for grids larger than the number of matrices
        for j in range(num_heatmaps, len(axes)):
            fig.delaxes(axes[j])

        plt.show()

def Confusion(y_true,y_pred, Title, figsize=(12, 8), Sub=False):
    """
    Plots confusion matrix/matrices for given true and predicted labels. 
    Supports both single and multiple confusion matrices as subplots.

    Inputs:
    - y_true (array-like or list of array-like): Ground truth (actual) labels. 
        - If 'Sub' is True, pass a list of label arrays for multiple confusion matrices.
    - y_pred (array-like or list of array-like): Predicted labels. 
        - If 'Sub' is True, pass a list of label arrays for multiple confusion matrices.
    - Title (str or list of str): Title(s) for the confusion matrix plot(s).
        - If 'Sub' is True, pass a list of titles corresponding to each confusion matrix.
    - figsize (tuple, optional): Figure size for the plot(s). Default is (12, 8).
    - Sub (bool, optional): If True, creates subplots for multiple confusion matrices. Default is False.
    """

    if not Sub:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        disp.ax_.set_title(Title,size=20)
    
    elif Sub:
    
        num_methods = len(Title)
        cols = 2
        rows = (num_methods + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for i in range(num_methods):
            cm = confusion_matrix(y_true[i], y_pred[i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            
            disp.plot(ax=axes[i], cmap='viridis', colorbar=False)
            axes[i].set_title(Title[i], fontsize=14)
            #axes[i].grid(False)  # Remove gridlines for a cleaner look
        
        #Hide extra subplots if num_methods < rows * cols
        for j in range(num_methods, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

def format_seconds(seconds):
    """
    Formats runtime seconds to a more readable format. 

    Inputs:
    - seconds (float): Runtime in seconds

    Outputs:
    - No output, prints total amount of seconds in format 2m 41s, for example. 
    """  
    minutes = seconds // 60
    seconds = seconds % 60
    return minutes, seconds

def extract_all_precisions(Reports, Labels):
    """
    Extracts precision values for each class from multiple classification reports.

    Inputs:
    - Reports: List of classification report dictionaries.
    - Labels: List of model names corresponding to each report.

    Outputs:
    - DataFrame: A table showing precision for each class across the reports.
    """
    precision_data = {}

    for label, report in zip(Labels, Reports):
        # Extract precision values for classes 0–9
        precision_data[label] = [
            f"{report[str(cls)]['precision']:.3f}" for cls in range(10)
        ]

    # Create a DataFrame for better visualization
    precision_df = pd.DataFrame(precision_data, index=[str(cls) for cls in range(10)])
    precision_df.index.name = "Class"
    return precision_df
