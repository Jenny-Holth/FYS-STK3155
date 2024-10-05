"""
# -*- coding: utf-8 -*-
Created on Sat Oct  5 10:19:51 2024

@author: Alexander Schei


###The bootstrap function is based on code from Hjorth-Jensen, M. (2021) Applied data analysis and machine learning section 5.4:  
#https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff

"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from Classes import Linear_Regression
from sklearn import linear_model
from sklearn.utils import resample
import sklearn.model_selection as sklms
from Classes import Plotting

#Making the design matrix
def design_matrix(x,y,p):
  X = np.zeros((len(x),2))
  X[:,0] = x
  X[:,1] = y
  poly = PolynomialFeatures(p)
  model = poly.fit_transform(X)
  return model

#Franke function, test function for testing parameters
def FrankeFunction(x,y):
    term1 = 0.75 * np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


#Function for finding MSE and R2 error over degrees for OLS
def degreerror(x,y,z,max_deg, noise, command,lamda):
    Degrees = np.arange(1,max_deg+1)
    MSElist = []
    R2list = []

    if isinstance(noise, np.ndarray):
        for j in noise:
            frank = FrankeFunction(x, y) + j*np.random.normal(0, 1, x.shape)
            MSEdeg = np.zeros((max_deg, 2))
            R2deg = np.zeros((max_deg, 2))
            for i in range(max_deg):
                X = design_matrix(x,y,i+1)
                OLS = Linear_Regression(X, frank, i+1)
                MSEdeg[i] = OLS.MSE_calc(command,lamda)
                R2deg[i] = OLS.R2_calc(command,lamda)
            MSElist.append(MSEdeg)
            R2list.append(R2deg)

    else:
        
        MSEdeg = np.zeros((max_deg, 2))
        R2deg = np.zeros((max_deg, 2))
        for i in range(max_deg):
            X = design_matrix(x,y,i+1)
            OLS = Linear_Regression(X, z, i+1)
            MSEdeg[i] = OLS.MSE_calc(command,lamda)
            R2deg[i] = OLS.R2_calc(command,lamda)
        MSElist.append(MSEdeg)
        R2list.append(R2deg)

    return R2list, MSElist, Degrees

#Function for easier 3d Trisurf plotting with 1,2 or 4 plots
def plot_predictions(Model, x, y, z,p, methods, lamda_ridge, lamda_lasso, titles):
    X_train, X_test, z_train, z_test, Xscaled_test, Xscaled_train = Model.split_scale()
    z_predictions = []; plot_titles = []; x_vals = []; y_vals = []
    Plot = Plotting()

    if "Original" in methods:
        z_predictions.append(z)
        plot_titles.append(titles[0])
        x_vals.append(x)  
        y_vals.append(y)  
    
    if "OLS" in methods:
        X = design_matrix(x,y,p[0])
        Model = Linear_Regression(X,z,p[0])
        zpred_train_OLS, zpred_test_OLS = Model.predict(0) 
        z_predictions.append(zpred_train_OLS)
        plot_titles.append(titles[1])
        x_vals.append(Xscaled_train[:, 1])  
        y_vals.append(Xscaled_train[:, 2])
        
    
    if "Ridge" in methods:
        X = design_matrix(x,y,p[1])
        Model = Linear_Regression(X,z,p[1])
        zpred_train_rid, zpred_test_rid = Model.predict(lamda_ridge)
        z_predictions.append(zpred_train_rid)
        plot_titles.append(titles[2])
        x_vals.append(Xscaled_train[:, 1])  
        y_vals.append(Xscaled_train[:, 2])
         

    if "Lasso" in methods:
        X = design_matrix(x,y,p[2])
        Model = Linear_Regression(X,z,p[2])
        Betas, zpred_train_las, zpred_test_las = Model.Lasso(lamda_lasso)
        z_predictions.append(zpred_train_las)
        plot_titles.append(titles[3])
        x_vals.append(Xscaled_train[:, 1]) 
        y_vals.append(Xscaled_train[:, 2])
        

    if len(methods) == 1:
        Plot.pred_vs_Franke(x_vals[0], y_vals[0], z_predictions[0], "Single", plot_titles[0])
    elif len(methods) == 2:
        Plot.pred_vs_Franke([x_vals[0], x_vals[1]], [y_vals[0], y_vals[1]], z_predictions, "Both", plot_titles)
    else:
        Plot.pred_vs_Franke(x_vals, y_vals, z_predictions, "All", plot_titles)


#Calculating beta value for given interval and degree
def beta_deg(max_deg, x, y, z, command, lamda):
    Beta_list = []

    for i in range(max_deg):
        X = design_matrix(x, y, i+1)
        if command == 'OLS':
            OLS = Linear_Regression(X, z, i+1)
            Beta_list.append(OLS.Beta(0))

        if command == 'Ridge':
            Ridge = Linear_Regression(X, z, i+1)
            Beta_list.append(Ridge.Beta(lamda))

        if command == 'Lasso':
            Lasso = Linear_Regression(X, z, i+1)
            Betas, ypred_train, ypred_test = Lasso.Lasso(lamda)
            Beta_list.append(Betas)

    return Beta_list


###The bootstrap function is based on code from Hjorth-Jensen, M. (2021) Applied data analysis and machine learning section 5.4:  
#https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
def bootstrap(x, y, z, max_deg, n_boot, command, lamda):
    x_train, x_test, y_train, y_test, z_train, z_test = sklms.train_test_split(x, y, z, test_size=0.2, random_state=999)

    MSE_boot = np.zeros(max_deg)
    Var_boot = np.zeros(max_deg)
    Bias_boot = np.zeros(max_deg)
    #Reshape z_test
    #z_test = z_test.reshape(-1, 1)
    z_testt = np.zeros((z_test.shape[0],1)); z_testt[:,0] = z_test[:]
    z_test = z_testt


    #Bootstrapping for each degree til max_deg
    for p in range(max_deg):
        z_pred = np.empty((z_test.shape[0], n_boot))
        X_test = design_matrix(x_test, y_test, p)

        for i in range(n_boot):
            x_, y_, z_ = resample(x_train, y_train, z_train)
            X_ = design_matrix(x_, y_, p)

            #OLS
            if command == "predict" and lamda == 0:
                Beta = np.linalg.inv(np.transpose(X_) @ X_) @ np.transpose(X_) @ z_
                Beta = Beta[:,np.newaxis]
                z_pred[:, i] = (X_test @ Beta).ravel()

            #Ridge
            elif command == "predict" and lamda > 0:
                Beta = np.linalg.pinv(X_.T @ X_ + np.identity(X_.shape[1]) * lamda) @ X_.T @ z_
                z_pred[:, i] = (X_test @ Beta).ravel()

            #Lasso
            elif command == "Lasso":
                Lasso = linear_model.Lasso(alpha=lamda, tol=8e-5, fit_intercept=False, max_iter=10000)
                Lasso.fit(X_, z_)
                z_pred[:, i] = Lasso.predict(X_test)


        MSE_boot[p] = np.mean((z_test - z_pred) ** 2)
        Bias_boot[p] = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True)) ** 2)
        Var_boot[p] = np.mean(np.var(z_pred, axis=1))

    return MSE_boot, Var_boot, Bias_boot

#Saves bootstrap data to text file
def savebootdata(filename,data):

    with open(filename, "w") as txt_file:
        #Makee header
        txt_file.write("MSE Variance Bias\n")
        
        #Save the data
        for lineMSE, lineVar, lineBias in zip(data[0],data[1],data[2]):

            txt_file.write(f"{lineMSE}  {lineVar} {lineBias} \n")
            

#Loads bootstrap data from text file
def loadbootdata(filename):

    MSE = []
    Variance = []
    Bias = []

    with open(filename, "r") as txt_file:
        # Skip the header
        next(txt_file)
        
        # Read each line and extract the values
        for line in txt_file:
            # Split the line into components (MSE, Variance, Bias)
            mse_val, var_val, bias_val = line.split()
            MSE.append(float(mse_val))
            Variance.append(float(var_val))
            Bias.append(float(bias_val))
    
    return MSE, Variance, Bias

#Preforms k-Fold cross validation for OLS, Ridge and Lasso
def kFold(xx, yy, zz, kk, degrees__, command, lamda):
    kfold = sklms.KFold(n_splits=kk)
    kfold.get_n_splits(xx)
    scores_kFold = np.zeros((len(degrees__), len(lamda)) if command != "OLS" else len(degrees__))
    
    for j in range(max(degrees__)):
        fold_mse = []

        for train_i, test_i in kfold.split(xx):
            xtrain, ytrain, ztrain = xx[train_i], yy[train_i], zz[train_i, np.newaxis]
            xtest, ytest, ztest = xx[test_i], yy[test_i], zz[test_i, np.newaxis]

            X_train = design_matrix(xtrain, ytrain, j+1)
            X_test = design_matrix(xtest, ytest, j+1)

            if command == "OLS":
                Beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ ztrain
                zpred_test = X_test @ Beta
                mse_test = np.mean((ztest - zpred_test)**2)
                fold_mse.append(mse_test)

            elif command == "Ridge":
                for i, lam in enumerate(lamda):
                    Beta = np.linalg.pinv(X_train.T @ X_train + np.identity(X_train.shape[1]) * lam) @ X_train.T @ ztrain
                    zpred_test = X_test @ Beta
                    mse_test = np.mean((ztest - zpred_test)**2)
                    scores_kFold[j, i] += mse_test / kk

            elif command == "Lasso":
                for i, lam in enumerate(lamda):
                    Lasso = linear_model.Lasso(alpha=lam, tol=8e-5, fit_intercept=False, max_iter=10000)
                    Lasso.fit(X_train[:, 1:], ztrain)
                    zpred_test = Lasso.predict(X_test[:, 1:])
                    mse_test = np.mean((ztest - zpred_test)**2)
                    scores_kFold[j, i] += mse_test / kk

        if command == "OLS":
            scores_kFold[j] = np.mean(fold_mse)

    if command == "OLS":
        return scores_kFold

    lamdaindex = np.argmin(scores_kFold, axis=1)
    return scores_kFold, lamdaindex


#Saves k-fold data to text file
def savekfolddata(filename,data,lamda,lamdaindex,degrees):

    with open(filename, "w") as txt_file:

        degrees_header = " ".join([f"PolynomialDegree{n}" for n in degrees])

        txt_file.write(f"LambdaIndex LambdaValue {degrees_header} \n")
        

        for i, (lam_val, lam_idx) in enumerate(zip(lamda, lamdaindex)):

            if (len(lamda)==1):
                txt_file.write(f"{lam_idx} {lam_val:.8e} ")
            else:
                txt_file.write(f"{lam_idx} {lamda[lam_idx]:.8e} ")
            
            if (len(lamda)==1):
                kfold_values_str = " ".join(map(str, data))
            else:
                kfold_values_str = " ".join(map(str, data[:, i]))
            
            txt_file.write(kfold_values_str + "\n")
            
#Loads k-fold data from text file
def loadkfolddata(filename):

    with open(filename, "r") as txt_file:
        header = txt_file.readline().strip().split()
        degrees = [int(deg.replace("PolynomialDegree", "")) for deg in header[2:]]
        
        lamdaindex = []
        lamda = []
        kfold_matrix = []

        for line in txt_file:
            parts = line.split()
            
            lamdaindex.append(int(float(parts[0])))
            lamda.append(float(parts[1]))
            
            kfold_values = list(map(float, parts[2:]))
            kfold_matrix.append(kfold_values)
    
    kfold_matrix = np.array(kfold_matrix).T
    
    return lamdaindex, lamda, degrees, kfold_matrix

#Finds best parameters for k-fold matrix of ridge and lasso
def find_best_hyperparameters(kfold_matrix, degrees, lamda):
    min_mse = np.min(kfold_matrix)
    min_index = np.unravel_index(np.argmin(kfold_matrix, axis=None), kfold_matrix.shape)
    
    best_degree = degrees[min_index[0]]
    best_lambda = lamda[min_index[1]]
    
    return best_degree, best_lambda, min_mse, min_index[1]

