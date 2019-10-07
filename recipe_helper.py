import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import functools

from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed

import pdb

class Recipe_Helper():
    def __init__(self, **params):
        self.X, self.y = None, None
        return

    def gen_data(self, num=30 , v=4,  a=0):
        """
        Generate a dataset of independent (X) an dependent (Y)

        Parameters
        -----------
        num: Integer.  The number of observations

        Returns
        --------
        (X,y): a tuple consisting of X and y.  Both are ndarrays
        """
        rng = np.random.RandomState(42)


        # X = num * rng.uniform(size=num)
        X = num * rng.normal(size=num)
        # X = X - X.min()

        X = X.reshape(-1,1)

        e = (v + a*X)
        y = v * X #  +  e * rng.uniform(-1,1, size=(num,1))

        a_term =  0.5 * a * (X**2)
        y = y + a_term

        return X,y

    def gen_plot(self, X,y, xlabel, ylabel):
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)


        _ = ax.scatter(X, y, color="red")


        _ = ax.set_xlabel(xlabel)
        _ = ax.set_ylabel(ylabel)

        return fig, ax


    def split(self, X,y, shuffle=True, pct=.80, seed=42):
        """
        Split the X and y datasets into two pieces (train and test)

        Parameters
        ----------
        X, y: ndarray.

        pct: Float.  Fraction (between 0 and 1) of data to assign to train
        seed: Float.  Seed for the random number generator

        Returns
        -------
        Tuple of length 4: X_train, X_test, y_train, y_test
        """
        # Random seed
        rng = np.random.RandomState(42)

        # Number of observations
        num = y.shape[0]

        # Enumerate index of each data point  
        idxs = list( range(0, num))

        # Shuffle indices
        if(shuffle):
            rng.shuffle(idxs)

        # How many observations for training ?
        split_idx = int( num * pct)

        # Split X and Y into train and test sets
        X_train, y_train = X[ idxs[:split_idx] ] , y[ idxs[:split_idx] ]
        X_test,  y_test  = X[ idxs[split_idx:] ],  y[ idxs[split_idx:] ]

        return X_train, X_test, y_train, y_test

    def plot_fit(self, X, y, ax=None,  on_idx=0):
        """
        Plot the fit

        Parameters
        ----------
        X: ndarray of features
        y: ndarray of targets
        ax: a matplotlib axes pbject (matplotlib.axes._subplots.AxesSubplot)

        Optional
        --------
        on_idx: Integer.  Which column of X to use for the horizontal axis of the plot

        """
        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(1,1,1)
            
        sort_idx = X[:, on_idx].argsort()
        X_sorted = X[ sort_idx,:]
        y_sorted = y[ sort_idx,:]

        _ = ax.plot(X_sorted[:, on_idx] , y_sorted, color="red")

    def transform(self, X):
        """
        Add a column to X with squared values

        Parameters
        ----------
        X: ndarray of features
        """
        X_p2 = np.concatenate( [X, X **2], axis=1)
        return X_p2

    def run_regress(self, X,y, run_transforms=False, plot_train=True,  xlabel=None, ylabel=None):
        """
        Do the full pipeline of the regression of y on X

        Parameters
        ----------
        X: ndarray of features
        y: ndarray of targets

        Optional
        --------
        runTransforms: Boolean.  If True, run additional data transformations to create new features
        """
        self.X, self.y = X, y

        # Split into train, test
        X_train, X_test, y_train, y_test = self.split(X,y)

        # Transform X's
        if (run_transforms):
            X_train = self.transform(X_train)
            X_test  = self.transform(X_test)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        

        # Create linear regression object
        regr = linear_model.LinearRegression()

        self.model = regr
        
        # Train the model using the training sets
        _ = regr.fit(X_train, y_train)

        # The coefficients
        print('Coefficients: \n', regr.intercept_, regr.coef_)
        # Lots of predictions: predict on entire test set
        y_pred = regr.predict(X_test)

        # Explained variance score: 1 is perfect prediction
        rmse_test = np.sqrt( mean_squared_error(y_test,  y_pred))

        print("\n")
        print("R-squared (test): {:.2f}".format(r2_score(y_test, y_pred)) )
        print("Root Mean squared error (test): {:.2f}".format( rmse_test ) )

        y_pred_train = regr.predict(X_train)

        rmse_train = np.sqrt( mean_squared_error(y_train,  y_pred_train))
        print("\n")
        print("R-squared (train): {:.2f}".format(r2_score(y_train, y_pred_train)) )
        print("Root Mean squared error (train): {:.2f}".format( rmse_train ) )

        # Plot predicted ylabel (red) and true label (black)
        num_plots = 2
        fig, axs = plt.subplots(1,num_plots, figsize=(12,4))
        
        _ = axs[0].scatter(X_test[:,0], y_test, color='black')
        _ = axs[0].scatter(X_test[:,0], y_pred, color="red")

        self. plot_fit(X_test, y_pred, ax=axs[0], on_idx=0)
        if xlabel is not None:
            _ = axs[0].set_xlabel(xlabel)

        if ylabel is not None:
            _ = axs[0].set_ylabel(ylabel)

        axs[0].set_title("Test")

        
        # Plot train
        if plot_train:
            _ = axs[1].scatter(X_train[:,0], y_train, color='black')
            _ = axs[1].scatter(X_train[:,0], y_pred_train, color="red")

            self. plot_fit(X_train, y_pred_train, ax=axs[1], on_idx=0)
            if xlabel is not None:
                _ = axs[1].set_xlabel(xlabel)

            if ylabel is not None:
                _ = axs[1].set_ylabel(ylabel)

            axs[1].set_title("Train")

        return

    def regress_with_error(self, X,y, run_transforms=False, plot_train=True,  xlabel=None, ylabel=None):
        # Run the regression.  Sets attributes of self
        self.run_regress(X, y, run_transforms, plot_train=plot_train, xlabel=xlabel, ylabel=ylabel)

        # Extract the results of running the split and regression steps
        X_train, X_test, y_train, y_test, regr = self.X_train, self.X_test, self.y_train, self.y_test, self.model

        y_pred = regr.predict(X_test)
        y_pred_train = regr.predict(X_train)

        fig, axs = plt.subplots(1,2, figsize=(12,4))

        # Plots for both test and train datasets
        for i, spec in enumerate( [ ("test", X_test, y_test, y_pred), ("train", X_train, y_train, y_pred_train) ] ):
            label, x, target, pred = spec
            ax = axs[i]
            # _= ax.scatter(x, target - pred, label="Error")
            _= ax.bar(x.reshape(-1), (target - pred).reshape(-1), label="Error")
            _ = ax.set_xlabel("Error")
            _ = ax.set_ylabel(xlabel)
            _ = ax.set_xlabel(ylabel)
            _= ax.set_title(label + " Error")
            _= ax.legend()

            return fig, axs
        
    def plot_resid(self, X, y, y_pred):
        resid_curve = y - y_pred
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.scatter(X, resid_curve)
        _ = ax.set_xlabel(xlabel)
        _ = ax.set_ylabel("Residual")
