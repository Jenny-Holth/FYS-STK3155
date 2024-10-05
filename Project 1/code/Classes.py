# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:18:33 2024

@author: Alexander Schei
"""

import numpy as np
import sklearn.model_selection as sklms
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from imageio import imread
import warnings
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
warnings.filterwarnings('ignore')


#Class for doing linear regression, finding MSE, R^2 and beta parameters
class Linear_Regression:
    def __init__(self, X, z, p):
        self.X = X; self.z = z; self.p = p
        self.X_train, self.X_test, self.z_train, self.z_test, self.Xscaled_test, self.Xscaled_train = self.split_scale()

    def split_scale(self):
        X = self.X
        z = self.z
        p = X.shape[1]
        X_train, X_test, z_train, z_test = sklms.train_test_split(X, z, test_size = 0.2, random_state=123)
        meanvalues_train = np.zeros(p)
        std_train = np.zeros(p)
        for i in range(p-1):
            meanvalues_train[i+1] = np.mean(X_train[:,i+1])
            std_train[i+1] = np.std(X_train[:,i+1])

        Xscaled_test = np.ones_like(X_test)
        Xscaled_train = np.ones_like(X_train)
        Xscaled_test[:,1:] = (X_test[:,1:] - meanvalues_train[1:])/std_train[1:]
        Xscaled_train[:,1:] = (X_train[:,1:] - meanvalues_train[1:])/std_train[1:]
        
        return X_train, X_test, z_train, z_test, Xscaled_test, Xscaled_train


    def Beta(self, lamda):
        #OLS
        if lamda == 0:
            Beta = np.linalg.inv(self.Xscaled_train.T @ self.Xscaled_train) @ self.Xscaled_train.T @ self.z_train
        #Ridge
        else:
            Beta = np.linalg.pinv(self.Xscaled_train.T @ self.Xscaled_train + np.identity(self.Xscaled_train.shape[1]) * lamda) @ self.Xscaled_train.T @ self.z_train
        
        return Beta

    def predict(self, lamda):
        X_train, X_test, z_train, z_test, Xscaled_test, Xscaled_train = self.split_scale()
        Beta = self.Beta(lamda)
        zpred_train = self.Xscaled_train @ Beta
        zpred_test = self.Xscaled_test @ Beta
        return zpred_train, zpred_test
    
    def Lasso(self, lamda): # Lasso regression from scikit-learn
        Lasso = linear_model.Lasso(lamda,tol=9e-2, max_iter=10000)
        Lasso.fit(self.Xscaled_train, self.z_train)
        Betas = Lasso.coef_
        zpred_train = Lasso.predict(self.Xscaled_train)
        zpred_test = Lasso.predict(self.Xscaled_test)
        return Betas, zpred_train, zpred_test

    def MSE(self, z, zpred):
        return np.mean((z - zpred) ** 2)

    def MSE_calc(self, command, lamda):
        if (command =="predict"):
            zpred_train, zpred_test = self.predict(lamda)
        else:
            _, zpred_train, zpred_test = self.Lasso(lamda)

        MSE_train = self.MSE(self.z_train, zpred_train)
        MSE_test = self.MSE(self.z_test, zpred_test)

        return MSE_train, MSE_test

    def R2(self, z, zpred):
        return 1 - np.sum((z-zpred)**2)/(np.sum((z-np.mean(z))**2))
    
    def R2_calc(self, command,lamda):
        if (command =="predict"):
            zpred_train, zpred_test = self.predict(lamda)
        else:
            _, zpred_train, zpred_test = self.Lasso(lamda)

        R2_train = self.R2(self.z_train, zpred_train)
        R2_test = self.R2(self.z_test, zpred_test)
        return R2_train, R2_test


#Class for all types of plotting
class Plotting:
    def __init__(self):
        self = self

    def plot_surface(self, ax, x, y, z, title, cmap=cm.coolwarm):
        surf = ax.plot_trisurf(x, y, z, linewidth=0, antialiased=True, cmap=cmap)
        ax.set_title(title, size=20)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        return surf

    def pred_vs_Franke(self, x, y, z, command, titles):
            plt.style.use('default')

            if command == "Single":
                fig = plt.figure(figsize=(13, 6))
                ax = fig.add_subplot(1, 2, 1, projection="3d")
                surf = self.plot_surface(ax, x, y, z, titles)
                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.show()

            elif command == "Both":
                fig = plt.figure(figsize=(13, 6))
                ax1 = fig.add_subplot(1, 2, 1, projection="3d")
                surf1 = self.plot_surface(ax1, x[0], y[0], z[0], titles[0])
                fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

                ax2 = fig.add_subplot(1, 2, 2, projection="3d")
                surf2 = self.plot_surface(ax2, x[1], y[1], z[1], titles[1])
                fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
                plt.show()

            elif command == "All":
                fig = plt.figure(figsize=(13, 6))
                ax = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

                for i in range(4):
                    surf = self.plot_surface(ax[i], x[i], y[i], z[i], titles[i])
                    fig.colorbar(surf, ax=ax[i], shrink=0.5, aspect=5)

                plt.tight_layout()
                plt.show()


    def MSE_R2(self, polydegree, RR, MSEE, noise, command, title):
        plt.style.use('ggplot')

        if isinstance(noise, (int, float)):
            plt.figure(figsize=(9, 6))
            plt.suptitle(title, size=20)

            if command == "Both":
                plt.subplot(2, 1, 1)
                plt.plot(polydegree, MSEE[:, 1], label='$MSE_{Test}$', linewidth=3)
                plt.plot(polydegree, MSEE[:, 0], label='$MSE_{Train}$', linewidth=3)
                plt.ylabel('MSE', size=20)
                plt.grid(True)
                plt.legend(fontsize=15)

                plt.subplot(2, 1, 2)
                plt.plot(polydegree, RR[:, 1], label='$R^2_{Test}$', linewidth=3)
                plt.plot(polydegree, RR[:, 0], label='$R^2_{Train}$', linewidth=3)
                plt.xlabel('Polynomial Degree', size=20)
                plt.ylabel('$R^2$', size=20)
                plt.grid(True)
                plt.legend(fontsize=15)

            else:
                plt.plot(polydegree, MSEE[:, 1], label='$MSE_{Test}$', linewidth=3)
                plt.plot(polydegree, MSEE[:, 0], label='$MSE_{Train}$', linewidth=3)
                plt.xlabel('Polynomial Degree', size=20)
                plt.ylabel('MSE', size=20)
                plt.grid(True)
                plt.legend(fontsize=15)

            plt.show()


        else:
            fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            plt.suptitle(title, size=20)

            for i in range(len(noise)):
                axs[0].plot(polydegree, MSEE[i][:, 1], label=f'Noise = {noise[i]}', linewidth=3)
                axs[0].axhline(y=0, color='k', linestyle='-')
                axs[0].set_ylabel('MSE', size=20)
                axs[0].grid(True)
                axs[0].legend(fontsize=15)

                axs[1].plot(polydegree, RR[i][:, 1], label=f'Noise = {noise[i]}', linewidth=3)
                axs[1].axhline(y=0, color='k', linestyle='-')
                axs[1].set_xlabel('Polynomial Degree', size=20)
                axs[1].set_ylabel('$R^2$', size=20)
                axs[1].grid(True)
                axs[1].legend(fontsize=15, loc='upper right')

            plt.tight_layout()
            plt.show()


    def lambda_Ridge_Lasso(self, RR, MSEE, lamdaa,title):
        plt.style.use('ggplot')
        plt.figure(figsize=(9, 6))
        plt.suptitle(title, size=20)
        
        plt.subplot(2,1,1)
        plt.plot(lamdaa, MSEE[:, 1], label='$MSE_{Test}$', linewidth=3)
        plt.plot(lamdaa, MSEE[:, 0], label='$MSE_{Train}$', linewidth=3)
        plt.xscale("log")
        plt.ylabel('MSE', size=20)
        plt.grid(True)
        plt.legend(fontsize=15, loc="upper right")

        plt.subplot(2,1,2)
        plt.plot(lamdaa, RR[:, 1], label='$R^2_{Test}$', linewidth=3)
        plt.plot(lamdaa, RR[:, 0], label='$R^2_{Train}$', linewidth=3)
        plt.xscale("log")
        plt.xlabel(r'$\lambda$', size=20)
        plt.ylabel('$R^2$', size=20)
        plt.grid(True)
        plt.legend(fontsize=15, loc="upper right")

        plt.show()


    def beta_deg(self,betalist, betamin, betamax, title):
        len_maxdeg = max(len(i) for i in betalist)
        betas = [[] for _ in range(len_maxdeg)]
        degrees = np.arange(len(betalist))
        plt.style.use('ggplot')

        for polydeg in betalist:
            for i in range(len(polydeg)):
                betas[i].append(polydeg[i])
            for i in range(len(polydeg), len_maxdeg):
                betas[i].append(None)

        plt.figure(figsize=(9, 6))
        for i in range(betamin, betamax):
            plt.plot(degrees, betas[i], label=rf'$\beta_{{{i}}}$', linewidth=2)

        plt.title(fr'$\beta$ as a function of poly-degree with {title}', size=18)
        plt.xlabel('Polynomial degree', size=15)
        plt.ylabel(r'$\beta$-value', size=15)
        plt.legend(fontsize=15, loc="lower left")
        plt.grid(True)
        plt.show()


    def bootstrap(self,MSE,Var,Bias,degree_,title):
        plt.style.use('ggplot')
        plt.figure(figsize=(9,6))
        plt.title(f'Bias-variance tradeoff with {title}',size=18)
        plt.plot(degree_, Var, label = 'Variance')
        plt.plot(degree_, Bias, label = 'Bias')
        plt.plot(degree_, MSE, color = 'black', linestyle = '--', label = 'MSE')
        plt.ylabel('Error',size=15)
        plt.xlabel('Polynomial degree',size=15)
        plt.grid(True)
        plt.legend(fontsize=15,loc='upper right')
        plt.show()


    def kfold_vs_boot(self,degreess,mse_fold,mse_boot,mse_org,kk,nn,title):
        plt.style.use('ggplot')
        plt.figure(figsize=(9, 6))
        plt.title(title, size=18)
        plt.plot(degreess, mse_fold, label=f"kFold (k={kk})",color="b", linewidth=2)
        plt.plot(degreess, mse_boot, label=f"Bootstrap (n={nn})",color="g", linewidth=2)
        if isinstance(mse_org, int):
            plt.xlabel("Degrees", size=15)
            plt.ylabel("MSE-Error", size=15)
            plt.legend(fontsize=15)
            plt.grid(True)
            plt.show()
        else:
            plt.plot(degreess, mse_org, label=f"MSE before resampling", color="r",linewidth=2)
            plt.xlabel("Degrees", size=15)
            plt.ylabel("MSE-Error", size=15)
            plt.legend(fontsize=15)
            plt.grid(True)
            plt.show()
    
    def heatmap(self,ridge, lasso, Lamda, degree):
        tix = [f'{float(text):1.1e}' for text in Lamda]

        plt.figure(figsize=(9,6))
        ax = sns.heatmap(ridge,linewidth=.5,xticklabels=tix,yticklabels=degree,annot=True)
        ax.set_title("MSE Heatmap with Ridge",size=20)
        ax.set_xlabel(r"$\lambda$",size=15)
        ax.set_ylabel("Polynomial Degree",size=15)
        plt.show()
        
        plt.figure(figsize=(9,6))
        ax = sns.heatmap(lasso,linewidth=.5,xticklabels=tix,yticklabels=degree,annot=True) 
        ax.set_title("MSE Heatmap with Lasso",size=20)
        ax.set_xlabel(r"$\lambda$",size=15)
        ax.set_ylabel("Polynomial Degree",size=15)
        plt.show()