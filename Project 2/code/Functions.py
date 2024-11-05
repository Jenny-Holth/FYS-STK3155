"""
Code is made using Python version 3.9.20. 
We have used Python libraries numpy, scikit-learn, matplotlib,  seaborn, autograd and numba. 
Code built on other work is cited above relevant classes/functions.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import random
import matplotlib.pyplot as plt
import autograd.numpy as np
import seaborn as sns
from sklearn.neural_network import MLPRegressor,  MLPClassifier
from sklearn.linear_model import LogisticRegression
from numba import njit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Classes import initialize_classes , NeuralNetwork, Gradient
from Utilities import sigmoid, sigmoid_der, cross_entropy, cross_entropy_der , ReLU, ReLU_der,leakyReLU,leakyReLU_der,linear,linear_der, R2, mse,mse_der, cost_func, cross_entropy, cross_entropy_der

np.random.seed(2024)
random.seed(2024)

def accuracy(prediction, target):
    """
    Calculates the accuracy between predictions and target values.

    Inputs:
    -prediction (array): Predicted values.
    -target (array): Target values.

    Outputs:
    -accuracy (float): Correctly predicted ratio.
    """
    prediction = np.where(prediction>=0.5,1,0)
    return np.sum(prediction == target) / len(target)

def check_accuracy(predictions, targets):
    """
    Checks and calculates the accuracy between predictions and target values, handling both 1D and 2D arrays.

    Inputs:
    -predictions (array): Array of predicted values.
    -targets (array): Array of target values.

    Outputs:
    -accuracy (float): Proportion of correctly predicted values.
    """

    predictions = np.where(predictions >= 0.5, 1, 0)
    
    #If input is 2D,use first, or use as is
    if predictions.ndim == 2:
        predictions = predictions[:, 0]
    if targets.ndim == 2:
        targets = targets[:, 0]
    

    accuracy = np.sum(predictions == targets) / len(targets)
    return accuracy

def get_data(dataset_type="cancer", nn=1000, pp=3):
    """
    Generates data based on specified dataset type.

    Inputs:
    -dataset_type (str): Type of dataset to generate, either "cancer" for the breast cancer dataset or "polynomial" for polynomial.
    -nn (int): Number of samples for data generation, default is 1000.
    -pp (int): Polynomial degree for design matrix, default is 3.

    Outputs:
    -inputs (array): Generated input data.
    -targets (array): Target values to the inputs.
    """
    np.random.seed(2024)
    random.seed(2024)
    if dataset_type == "cancer":
        breast_cancer = load_breast_cancer()
        inputs1 = breast_cancer.data
        targets1 = breast_cancer.target.reshape(-1,1)
        return inputs1, targets1
    
    elif dataset_type == "polynomial":
        xx = np.random.rand(nn)
        Target = (1 + 2*xx + 3*xx**2) + 0.1* np.random.normal(0,1, xx.shape)
        Target = Target.reshape(-1,1)
        XX = design_matrix(xx, nn, pp)
        return XX, Target

def train_command(Inputs, Targets, Methods, Commands, Gradient_type, initial_conditions, Custom, MLP=False):
    """
    Trains a neural network using cancer data with specified methods and commands, and prints accuracy.
    Inputs:
    -Inputs (np.array): Input data for training.
    -Targets (np.array): Target labels for training.
    -Methods (list of str): Training methods to use, ["Normal", "Stochastic"].
    -Commands (list of str): Optimizers to use, ["GD", "AdaGrad", "Adam", "RMSprop"].
    -Gradient_type (str): Type of gradient calculation ("Normal","Back","AutoGrad").
    -initial_conditions (dict): Dictionary of parameters like eta, batch_size, max_iter, etc.
    -MLP (True/False): If True, function will incluce results from MLPClassifier, if False it will skip it 

    Output:
    -case_list (list): Predicted outputs for each command-method.
    -target_list (list): Target outputs for specific case.
    """ 
    np.random.seed(2024)
    random.seed(2024)
    if isinstance(Commands, str):
        Commands = [Commands]

    x_train, x_test, targets_train, targets_test = train_test_split(Inputs, Targets, test_size=0.30, random_state=123)
    case_list = []; case_list_MLP = []
    target_list = []; target_list_MLP = []

    #Loop through each method and command
    for Method in Methods:
        if Method == "Normal":
            etas = initial_conditions.get("eta_Normal")
        else: 
            etas = initial_conditions.get("eta_Stochastic") 

        for Command in Commands:

            #Initialize the neural network and gradient optimizer
            Neural_Net, Optimizer = initialize_classes(initial_conditions, x_train, targets_train, custom_conditions=Custom)
            #Select the right optimization method
            optimization_method = {
                "Normal": Optimizer.Optimize_Normal,
                "Stochastic": Optimizer.Optimize_Stochastic
            }[Method]

            layers = optimization_method(Eta=etas, lamda = Optimizer.lamda, command=Command, 
                gradient_type=Gradient_type)
            print(f'For {Method}, with {Command}, momentum = {Optimizer.use_momentum}:')

            if MLP and Method == "Stochastic":
                if Command == "GD":
                    solv = 'sgd'
                else:
                    solv = 'adam'
                if Command == "GD" or Command == "Adam":
                    if Optimizer.use_momentum:
                        dm = Optimizer.dm
                    else:
                        dm = 0
                    layer_output_sizes = Optimizer.layer_output_sizes
                    cancer_classify = MLPClassifier(hidden_layer_sizes=layer_output_sizes[:-2], solver=solv,
                                        activation = 'logistic',
                                        batch_size = Optimizer.batch_size,
                                        random_state = 123, 
                                        learning_rate_init = etas[Command] ,
                                        momentum = dm, 
                                        validation_fraction = 0, 
                                        max_iter = Optimizer.max_iter).fit(x_train, targets_train.ravel() )
                    y_pred = cancer_classify.predict(x_test)
                    prediction = (y_pred >= 0.5).astype(int)
                    acc_MLPC = np.sum(prediction==targets_test.ravel())/len(prediction)
                    print(f"Accuracy MLP = {acc_MLPC:.5f}")
                    case_list_MLP.append(prediction)
                    target_list_MLP.append(targets_test)

            case = Neural_Net.feed_forward_batch(x_test, layers)
            accuracy = check_accuracy(case, targets_test)

            print(f"Accuracy    = {accuracy:.5f}")

            case_list.append((case >= 0.5).astype(int))
            target_list.append(targets_test)

    if not MLP:
        return case_list , target_list
    else:
        return case_list , target_list 

def plot_confusion_matrices(Predictions,Targets,Title, Filename, Optimizer=["GD"]):
    """
    Plots confusion matrices for soptimizers with Method.
    Inputs:
    -Predictions (dict/array): If dict, should contain keys matching Optimizer list, with predicted values as arrays.
                                If array, should be the predictions to a single optaimizer.
    -Targets (dict/array): If dict, should contain keys matching Optimizer list, with values as arrays.
                            If array, should be the values to a single optimizer.
    -Method (str): The method used "Normal" or "Stochastic".
    -Optimizer (list): List of optimizers to plot ["GD", "AdaGrad", "Adam", "RMSprop"].
    """
    
    num_optimizers = len(Optimizer)
    
    #Make layout based on number of optimizers
    if num_optimizers == 1:

        name = f"confu_{Filename}_{Optimizer[0]}.png"
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cm = confusion_matrix(
            Targets if not isinstance(Targets, dict) else Targets[Optimizer[0]], 
            Predictions if not isinstance(Predictions, dict) else Predictions[Optimizer[0]])
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False, values_format=None)
        ax.set_title(f"{Title} - {Optimizer[0]}")

    else:
        #Set up a 2x2 grid for up to 4 optimizers
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()  # Flatten to make indexing easier
        fig.suptitle(f"{Title}",size=18)
        fig.subplots_adjust(top=0.88)
        name = f"confu_{Filename}_all.png"
        for idx, opt in enumerate(Optimizer):
            cm = confusion_matrix(Targets[opt], Predictions[opt])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=axs[idx], colorbar=False)
            axs[idx].set_title(f"{opt}",size=15)
        
        # Hide any unused subplots if Optimizer length < 4
        for idx in range(num_optimizers, 4):
            axs[idx].axis("off")
    plt.savefig(name)
    plt.tight_layout()
    plt.show()

def pred_target_dict(Commands, Case_list, Target_list):
    """
    Makes prediction and target dictionaries from Commands, Case_list, and Target_list.
    Inputs:
    -Commands (str or list): List of optimizers or a single optimizer as a string.
    -Case_list (list or array-like): List of predictions for each optimizer, or a single prediction array.
    -Target_list (list or array-like): List of targets for each optimizer, or a single target array.

    Outputs:
    -Predictions_dict (dict): Dictionary of predictions with optimizers as keys.
    -Target_dict (dict): Dictionary of targets with optimizers as keys.
    """
    if isinstance(Commands, str):
        Commands = [Commands]
    if not isinstance(Case_list, list):
        Case_list = [Case_list]
    if not isinstance(Target_list, list):
        Target_list = [Target_list]

    # Create dictionaries
    Predictions_dict = {cmd: Case_list[idx] for idx, cmd in enumerate(Commands)}
    Target_dict = {cmd: Target_list[idx] for idx, cmd in enumerate(Commands)}
    
    return Predictions_dict, Target_dict

def kFold(k, Method, Command, initial_conditions, Custom = None, Data_type = "cancer", Gradient_type="Back", Scale = True, MLP=False, sklearn = False):
    """
    Performs k-fold cross-validation for training and evaluating models with gradient-based optimizers or scikit-learn estimators.

    Inputs:
    -k (int): Number of folds in cross-validation.
    -Method (str): Optimization method ("Normal" or "Stochastic").
    -Command (str): Optimizer command ("GD", "AdaGrad", "Adam", "RMSprop").
    -initial_conditions (dict): Default conditions for network configuration and optimizer parameters.
    -Custom (dict, optional): Custom layer or neuron settings for the neural network.
    -Data_type (str, optional): Type of dataset ("cancer", "polynomial").
    -Gradient_type (str, optional): Gradient calculation type ("Back", "Normal", or "AutoGrad").
    -Scale (True/False, optional): If True, scales inputs.
    -MLP (True/False, optional): If True, uses MLPClassifier for training.
    -sklearn (True/False, optional): If True, uses LogisticRegression for training.

    Outputs:
    -case_list (array): Concatenated predictions from each fold.
    -target_list (array): Concatenated target values from each fold.
    """
    np.random.seed(2024)
    random.seed(2024)

    kfold = KFold(n_splits=k)
    Inputs, Targets = get_data(Data_type, nn=1000)
    if Scale:
        scaler = StandardScaler()
        Inputs = scaler.fit_transform(Inputs)

    if Method == "Normal":
        etas = initial_conditions.get("eta_Normal", 0)
    else: 
        etas = initial_conditions.get("eta_Stochastic", 0) 

    acc_list = [];  case_list = [];  target_list = []

    for train_i, test_i in kfold.split(Inputs):
        xtrain, ytrain = Inputs[train_i], Targets[train_i]
        xtest, ytest = Inputs[test_i], Targets[test_i]

        Neural_Net, Optimizer = initialize_classes(initial_conditions, xtrain, ytrain, custom_conditions=Custom)
        optimization_method = {
            "Normal": Optimizer.Optimize_Normal,
            "Stochastic": Optimizer.Optimize_Stochastic
        }[Method]

        if MLP and Method == "Stochastic":
            if Command == "GD":
                solv = 'sgd'
            else:
                solv = 'adam'
            if Command == "GD" or Command == "Adam":
                if Optimizer.use_momentum:
                    dm = Optimizer.dm
                else:
                    dm = 0
            if Command == "GD" or Command == "Adam":
                layer_output_sizes = Optimizer.layer_output_sizes
                cancer_classify = MLPClassifier(hidden_layer_sizes=layer_output_sizes[:-2], solver=solv,
                                    activation = 'logistic',
                                    batch_size = Optimizer.batch_size,
                                    random_state = 123, 
                                    learning_rate_init = etas[Command] ,
                                    momentum = dm, 
                                    validation_fraction = 0, 
                                    max_iter = Optimizer.max_iter).fit(xtrain, ytrain.ravel() )
                y_pred = cancer_classify.predict(xtest)
                prediction = (y_pred >= 0.5).astype(int)
                acc_MLPC = np.sum(prediction == ytest.ravel())/len(prediction)
                acc_list.append(acc_MLPC)
                target_list.append(ytest)
                case_list.append(prediction)
                if k< 10:
                    print(f"Accuracy MLP = {acc_MLPC:.5f}")

            else:
                print(f"Method {Method} or Command {Command} not supported for MLPClassifier")
                break

        else:
            
            if sklearn:
                cancer_classify = LogisticRegression().fit(xtrain, ytrain.ravel())
                y_pred = cancer_classify.predict(xtest)
                y_pred = (y_pred >= 0.5).astype(int)
                acc_LR = np.sum(y_pred == ytest.ravel())/len(y_pred)
                target_list.append(ytest)
                case_list.append(y_pred)
                acc_list.append(acc_LR)               
                if k< 10:
                    print(f"Accuracy from LR = {acc_LR:.4f}")

            
            else:

                layers = optimization_method(Eta=etas, lamda=0, command=Command, 
                                gradient_type=Gradient_type)
                #layers = Neural_Net.train_network(xtrain, ytrain, 0.1, 5000)
                
                cases = Neural_Net.feed_forward_batch(xtest, layers)
                cases = (cases >= 0.5).astype(int)
                case_list.append(cases)
                target_list.append(ytest)
                acc = check_accuracy(cases, ytest)
                acc_list.append(acc)
                if k< 10:
                    print(f"Accuracy = {acc:.4f}")


        
    if MLP or sklearn:
        print(f"For {Method} with {Command}")
        print(f"Mean acc from after {k}-folds = {np.mean(acc_list):.4}")
        case_list = np.concatenate([arr for arr in case_list])
        target_list = np.concatenate([arr for arr in target_list])
        return case_list, target_list

    else:
        print(f"For {Method} with {Command}")
        print(f"Mean acc after {k}-folds = {np.mean(acc_list):.4}")
        case_list = np.concatenate([arr.flatten() for arr in case_list])
        target_list = np.concatenate([arr.flatten() for arr in target_list])
        return np.array(case_list), np.array(target_list)

def kFold_multiple(k, Method, Commands, initial_conditions, Custom = None, Data_type = "cancer", Gradient_type="Back", Scale = True, MLP=False, sklearn = False):
    """
    Runs k-fold cross-validation for multiple optimizer commands and compiles results.

    Inputs:
    -k (int): Number of folds in cross-validation.
    -Method (str): Optimization method ("Normal" or "Stochastic").
    -Commands (list): List of optimizer commands ("GD", "AdaGrad", "Adam", "RMSprop").
    -initial_conditions (dict): Default conditions for network configuration and optimizer parameters.
    -Custom (dict, optional): Custom layer or neuron settings for the neural network.
    -Data_type (str, optional): Type of dataset ("cancer", "polynomial").
    -Gradient_type (str, optional): Gradient calculation type ("Back", "Normal", or "AutoGrad").
    -Scale (True/False, optional): If True, scales inputs.
    -MLP (True/False, optional): If True, uses MLPClassifier for training.
    -sklearn (True/False, optional): If True, uses LogisticRegression for training.

    Outputs:
    -super_case (list): List of concatenated predictions for each command.
    -super_target (list): List of concatenated true target values for each command.
    """

    super_case = []
    super_target = []
    for Command in Commands:
        case_list, target_list = kFold(k, Method=Method, Command=Command, 
                                       initial_conditions=initial_conditions, 
                                       Custom= Custom,
                                       Data_type = Data_type, Gradient_type = Gradient_type, 
                                       Scale = Scale, MLP = MLP, sklearn = sklearn)
        super_case.append(case_list)
        super_target.append(target_list)
    return super_case, super_target

def train_command_poly(Method, Command, Gradient_type, initial_conditions, Custom, Scale = False, MLP = False):
    """
    Trains models with polynomial data using specified optimizers and gradient types.

    Inputs:
    -Method (list): Optimization methods ("Normal", "Stochastic").
    -Command (list): Optimizer commands ("GD", "AdaGrad", "Adam", "RMSprop").
    -Gradient_type (str): Gradient calculation type ("Back", "Normal", "AutoGrad").
    -initial_conditions (dict): Default conditions for network configuration and optimizer parameters.
    -Custom (dict, optional): Custom layer or neuron settings for the neural network.
    -Scale (True/False, optional): If True, scales inputs.
    -MLP (True/False): If True, function will incluce results from MLPClassifier, if False it will skip it 

    Outputs:
    -results (list): Accuracy, MSE, or R2 values of each model based on Gradient_type.
    """
    np.random.seed(2024)
    random.seed(2024)
    scaler = StandardScaler()
    if Method == "Normal":
        etas = initial_conditions.get("eta_Normal")
    else: 
        etas = initial_conditions.get("eta_Stochastic") 

    results = []
    for i in Method:
        for j in Command:

            Inputs, Targets = get_data("polynomial", nn=1000)
            if Scale:
                Inputs = scaler.fit_transform(Inputs)

            x_train, x_test, targets_train, targets_test = train_test_split(Inputs, Targets, test_size=0.20, random_state=123)
            
            Neural_Net, Optimizer = initialize_classes(initial_conditions, x_train, targets_train, custom_conditions=Custom)

            optimize_method = {
            "Normal": Optimizer.Optimize_Normal,
            "Stochastic": Optimizer.Optimize_Stochastic}
            typ = optimize_method[i]

            layers = typ(Eta =etas ,lamda = Optimizer.lamda, 
                         command=j, gradient_type=Gradient_type)

            print(f'For {i}, with {j}, lambda =  {Optimizer.lamda} momentum = {Optimizer.use_momentum}:')

            if (MLP == True) and (i == "Stochastic"):
                if j == "GD":
                    solv = 'sgd'
                else:
                    solv = 'adam'

                if j == "GD" or j == "Adam":
                    if Optimizer.use_momentum:
                        dm = Optimizer.dm
                    else:
                        dm = 0

                    if Optimizer.activation_funcs[0] is sigmoid:
                        acti = 'logistic'
                    else:
                        acti = 'relu'


                    Targets = Targets.reshape(len(Targets))
                    x_train_s, x_test_s, targets_train_s, targets_test_s = train_test_split(Inputs, Targets, test_size=0.20, random_state=123)

                    layer_output_sizes = Optimizer.layer_output_sizes
                    poly_regression = MLPRegressor(hidden_layer_sizes=layer_output_sizes[:-1], 
                            batch_size = Optimizer.batch_size,
                            validation_fraction=0.0,
                            activation=acti,
                            solver=solv,
                            max_iter = Optimizer.max_iter,
                            momentum=dm).fit(x_train_s, targets_train_s)

                    y_pred = poly_regression.predict(x_test_s)
                    r2 = poly_regression.score(x_test_s, targets_test_s)
                    regMSE = mse(y_pred, targets_test_s)  
                    print(f"MSE-MLP = {regMSE:.4f} ")
                    print(f"R2-MLP  = {r2:.4f} ")


            if Gradient_type == "Back":
                case = Neural_Net.feed_forward_batch(x_test, layers)
                accuracy = mse(case, targets_test)
                r2 = R2(case, targets_test)
                print(f"MSE = {accuracy:.4f}")
                print(f"R2  = {r2:.4f} ")
                results.append([accuracy,r2])

            else:    

                zz = x_test@layers
                msee = np.mean((zz - targets_test) ** 2)
                r2 = R2(zz, targets_test)
                print(f"MSE = {msee:.4f}")
                print(f"R2  = {r2:.4f} ")
                results.append([msee,r2])

    return results

def design_matrix(x, n, p):
    """
    Generates a design matrix for polynomial regression.

    Inputs:
    -x (array): 1-dimensional input data array.
    -n (int): Number of rows in the design matrix.
    -p (int): Degree of the polynomial, representing the number of columns in the design matrix.

    Outputs:
    -X (array): Design matrix of shape (n, p).
    """
    X = np.zeros((n,p))
    for i in range(p):
      X[:,i] = np.transpose(x**i)
    return X

def heatmap(Matrix, Title, Filename, eta_list, lamda_list, LogScale = False):
    """
    Generates a heatmap to visualize mean squared error (MSE) values across different learning rates and regularization penalties.

    Inputs:
    -Matrix (2D array): The data matrix to plot, with MSE values.
    -Title (str): Title for the heatmap plot.
    -Filename (str): Filename for saving the heatmap image.
    -eta_list (list): List of learning rates for labeling the x-axis.
    -lamda_list (list): List of lambda values for labeling the y-axis.
    -LogScale (bool): If True, applies a log10 scale to the Matrix values.

    Outputs:
    -None: Displays and saves the heatmap as a PNG file with the specified filename.
    """

    tix_eta = [f'{float(text):1.1e}' for text in eta_list]
    tix_lamda = [f'{float(text):1.1e}' for text in lamda_list]
    if LogScale:
       Matrix = np.log10(Matrix)

    plt.figure(figsize=(9, 6))
    ax = sns.heatmap(Matrix,linewidth=.5,xticklabels=tix_eta,yticklabels=tix_lamda, annot=True)
    ax.collections[0].colorbar.set_label("MSE",size=15)
    ax.set_title(f"{Title}",size=20)
    ax.set_xlabel(r"Learning rate $\eta$",size=15)
    ax.set_ylabel(r"Penalty $\lambda$",size=15)
    plt.savefig(f"HeatETALamda_{Filename}.png")
    plt.show()

def accmatrix_layers_neurons(Inputs, Target, Method, Command, initial_conditions, Layers, Neurons, Gradient_type="Back"):
    """
    Generates an accuracy matrix for varying layer and neuron configurations.

    Inputs:
    -Inputs (array): Training input data.
    -Target (array): Target output values.
    -Method (str): Optimization method, either "Normal" or "Stochastic".
    -Command (str): Specific optimizer type, such as "GD", "AdaGrad", etc.
    -initial_conditions (dict): Dictionary of initial parameters and configurations.
    -Layers (list): List of layer configurations to test.
    -Neurons (list): List of neuron counts per layer.
    -Gradient_type (str): Gradient method, default is "Back".

    Outputs:
    -acc_matrix (2D array): Matrix of accuracy values (Layers, Neurons).
    """
    np.random.seed(2024)
    random.seed(2024)
    x_train, x_test, targets_train, targets_test = train_test_split(Inputs, Target, test_size=0.30, random_state=123)
    etas = initial_conditions.get("eta_Normal" if Method == "Normal" else "eta_Stochastic")

    acc_matrix = np.zeros((len(Neurons), len(Layers)))

    for m, num_neurons in enumerate(Neurons):

        for n, num_layers in enumerate(Layers):

            layer_sizes = [num_neurons] * num_layers + [1]  # Fixed neuron count across all layers
            activation_funcs = [sigmoid] * len(layer_sizes)
            activation_ders = [sigmoid_der] * len(layer_sizes)

            layers_config = {
                "layer_output_sizes": np.array(layer_sizes),
                "activation_funcs": activation_funcs,
                "activation_ders": activation_ders
            }
            x_train, x_test, targets_train, targets_test = train_test_split(Inputs, Target, test_size=0.30, random_state=123)
            etas = initial_conditions.get("eta_Normal" if Method == "Normal" else "eta_Stochastic")
            # Initialize classes
            Neural_Net, Optimizer = initialize_classes(initial_conditions, x_train, targets_train, custom_conditions=layers_config)
            
            # Select and run the optimizer
            optimization_method = {
                "Normal": Optimizer.Optimize_Normal,
                "Stochastic": Optimizer.Optimize_Stochastic
            }[Method]
            learning_rate = 0.1
            epochs = 3000
            #layers = optimization_method(Eta=etas, lamda=Optimizer.lamda, command=Command, gradient_type=Gradient_type)
            layers = Neural_Net.create_layers_batch(SqrtDivide = False)
            #layers = optimization_method(Eta=etas, lamda=Optimizer.lamda, command=Command, gradient_type=Gradient_type)

            for i in range(epochs):
                layers_grad = Neural_Net.backpropagation_batch(x_train, targets_train, layers)
                for (W,b),(W_g,b_g) in zip(layers,layers_grad):
                    W -= learning_rate * W_g
                    b -= learning_rate * b_g


            # Calculate accuracy and store in acc_matrix
            case = Neural_Net.feed_forward_batch(x_test, layers)
            acc_matrix[m, n] = check_accuracy(case, targets_test)

    return acc_matrix



def heatmap_acc(Matrix, Title, Filename, Layers, Neurons, LogScale=False):
    """
    Generates a heatmap to display accuracy values across different layer and neuron configurations.

    Inputs:
    -Matrix (2D array): The accuracy matrix to be plotted.
    -Title (str): Title for the heatmap plot.
    -Filename (str): Filename for saving the plot image.
    -Layers (list): List of layer configurations to label x-axis.
    -Neurons (list): List of neuron configurations to label y-axis.
    -LogScale (bool): If True, applies log10 scaling to Matrix values.

    Outputs:
    -None: Displays and saves the heatmap as a PNG file with the specified filename.

    Description:
    This function plots a heatmap of accuracy values based on the number of layers and neurons, enabling the visualization of how different neural network architectures affect model performance.
    """

    if LogScale:
       Matrix = np.log10(Matrix)
    plt.figure(figsize=(9, 6))
    ax = sns.heatmap(Matrix,linewidth=.5,xticklabels=Layers, yticklabels=Neurons, annot=True, fmt = ".3g")
    ax.collections[0].colorbar.set_label("Accuracy",size=15)
    ax.set_title(f"{Title}",size=20)
    ax.set_xlabel("Layers",size=15)
    ax.set_ylabel("Neurons",size=15)
    plt.savefig(f"HeatLayNeu_{Filename}.png")
    plt.show()
