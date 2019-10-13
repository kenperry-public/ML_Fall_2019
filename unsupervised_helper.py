import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pickle
import math

import os
import time

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

from sklearn.decomposition import PCA



import mnist_helper as mnhelp

class PCA_Helper():
    def __init__(self, **params):
        return

    def mnist_init(self):
        mnh = mnhelp.MNIST_Helper()
        self.mnh = mnh

        X, y = mnh = mnh.fetch_mnist_784()

        return X, y

    def mnist_PCA(self, X, n_components=0.95, **params):
        """
        Fit PCA to X

        Parameters
        ----------
        n_components: number of components
        - Passed through to sklearn PCA
        -- <1: interpreted as fraction of explained variance desired
        -- >=1 interpreted as number of components
        
        """
        if n_components is not None:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA()

        pca.fit(X)

        return pca

    def transform(self, X,  model):
        """
        Transform samples through sklearn model

        Parameters
        ----------
        X: ndarray (num_samples, num_features)
        model: sklearn model object, e.g, PCA

        X_reduced: ndarray (num_samples, pca.num_components_)
        """
        X_transformed = model.transform(X)
        return X_transformed

    def inverse_transform(self, X,  model):
        """
        Invert  samples that were transformed through sklearn model

        Parameters
        ----------
        X: ndarray (num_samples, num_features_trasnformed)
        model: sklearn model object, e.g, PCA

        X_reconstruct: ndarray (num_samples, num_features)
        """
        X_reconstruct = model.inverse_transform(X)
        return X_reconstruct

    def num_components_for_cum_variance(self, pca, thresh):
        """
        Return number of components of PCA such that cumulative variance explained exceeds threshhold

        Parameters
        ----------
        pca: PCA object
        thresh: float. Fraction of explained variance threshold
        """

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= thresh) + 1

        return d

    def plot_cum_variance(self, pca):
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        x  = range(1, 1 + cumsum.shape[0])
        
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        _ = ax.plot(x, cumsum)

        _ = ax.set_title("Cumulative variance explained")
        _ = ax.set_xlabel("# of components")
        _ = ax.set_ylabel("Fraction total variance")

        _= ax.set_yticks( np.linspace(0,1,11)  )

        return fig, ax

    def mnist_filter(self, X, y, digit):
        cond = (y==digit)
        X_filt = X[ cond ]
        y_filt = y[ cond ]

        return X_filt, y_filt

    def mnist_plot_2D(self, X, y):
        fig, ax = plt.subplots(1,1, figsize=(12,6))

        cmap="jet"
        _ = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
        # _ = ax.axis('off')

        _ = ax.set(xlabel='component 1', ylabel='component 2')

        
        norm = mpl.colors.Normalize(vmin=0,vmax=9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # _ = plt.colorbar(sm)

        return fig, ax


class VanderPlas():
    def __init__(self, **params):
        return

    """
    The following set of methods illustrate PCA and are drawn from 
    http://localhost:8888/notebooks/NYU/external/PythonDataScienceHandbook/notebooks/05.09-Principal-Component-Analysis.ipynb#Introducing-Principal-Component-Analysis
    """
    def create_data(self):
        rng = np.random.RandomState(1)
        X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

        rng = np.random.RandomState(1)
        X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

        return X


    def draw_vector(self, v0, v1, ax=None):
        arrowprops=dict(arrowstyle='->',
                        linewidth=2,
                        color="black",
                        shrinkA=0, shrinkB=0)
        _ = ax.annotate('', v1, v0, arrowprops=arrowprops)

        return ax

                   
    def show_2D(self, X, whiten=False, alpha=0.4):
        """
        Plot the dataset (X) and show the PC's, both in original feature space and transformed (PC) space

        Parameters
        ----------
        X: feature matrix

        whiten: Boolean,  whiten argument to PCA constructor 
        alpha: alpha for scatter plot (plt.scatter argument)
        """

        pca = PCA(n_components=2, whiten=whiten)
        pca.fit(X)

        print(pca.components_)
        print(pca.explained_variance_)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        _ = fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

        # plot data in original feature space
        _ = ax[0].scatter(X[:, 0], X[:, 1], alpha=alpha)

        # Show the components (i.e, PC's, axes) in original feature space
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)
            self.draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
        ax[0].axis('equal');
        # _ =ax[0].set(xlabel='x', ylabel='y', title='input')
        _= ax[0].set_title("Original")
        _= ax[0].set_xlabel("$x_1$")
        _= ax[0].set_ylabel("$x_2$")
        


        # plot data in transformed (PC) space
        X_pca = pca.transform(X)
        _ = ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=alpha)

        # Show the axes in transformed (i.e, rotated) feature space
        if whiten:
            self.draw_vector([0, 0], [0, 3], ax=ax[1])
            self.draw_vector([0, 0], [3, 0], ax=ax[1])
        else:
            for length, vector in zip(pca.explained_variance_, [np.array([-1,0]), np.array([0,1]) ] ):
                v = vector * 3 * np.sqrt(length)
                self.draw_vector([0,0], [0,0] + v, ax=ax[1])

        ax[1].axis('equal')
        _ = ax[1].set(xlabel='component 1', ylabel='component 2',
                  title='principal components',
                  # xlim=(-5, 5), ylim=(-3, 3.1)
                     )

        fig.tight_layout()

    """
    The following methods show dimensionality reduction using PCA on sklearns "digits" dataset (low resolution digits).
    It is derived from:
    http://localhost:8888/notebooks/NYU/external/PythonDataScienceHandbook/notebooks/05.09-Principal-Component-Analysis.ipynb#PCA-for-visualization:-Hand-written-digits
    """

    def digits_plot(self, data):
        """
        Plot the data from the digits dataset (each digit is 8x8 matrix of pixels)
        
        Parameters
        ----------
        data: List.  Elements are digits (8x8 matrices)
        """
        fig, axes = plt.subplots(4, 10, figsize=(12, 6),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))

        for i, ax in enumerate(axes.flat):
            _ = ax.imshow(data[i].reshape(8, 8),
                      cmap='binary', interpolation='nearest',
                      clim=(0, 16))

        return fig, axes

    def digits_reconstruction(self, data, n_components=None):
        """
        Transform data via PCA using num_components PC's, and show reconstruction
        """
        if n_components is not None:
            pca = PCA(n_components)  # project from 64 to 2 dimensions
        else:
            pca = PCA()
        
        projected = pca.fit_transform(data)

        print(data.shape)
        print(projected.shape)

        reconstructed = pca.inverse_transform(projected)
        _ = self.digits_plot(reconstructed)

        return pca, projected, reconstructed

    def digits_show_clustering(self, projected, targets, alpha=0.9):
        plt.figure(figsize=(10,6))

        plt.scatter(projected[:, 0], projected[:, 1],
                    c=targets, edgecolor='none', alpha=alpha,
                    cmap=plt.cm.get_cmap('Accent', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar();


    def plot_cum_variance(self, pca):
        pch = PCA_Helper()
        return pch.plot_cum_variance(pca)

class YieldCurve_PCA():
    """
    Perform PCA on CHANGES in the  yield curve.

    Derived from:
    https://github.com/radmerti/MVA2-PCA/blob/master/YieldCurvePCA.ipynb

    NOTES
    -----
    - The input data seems to be monthly observations of the level of the Yield Curve
    - Another resource does a PCA of the swap yield curve
    https://clinthoward.github.io/portfolio/2017/08/19/Rates-Simulations/

    -- fewer maturities
    -- But gets data from Quandl
    --- Uses FRED so can also get from the via pandas datareader.
    --- Better choice than the unknown csv file above, but would require additional modules

    - Both of the above notebooks do PCA on the LEVEL of the yield curve, NOT the change

    - I have modified it for changes in level.  That is more appropriate to risk management and is in keeping with the
    -- original Litterman Scheinkman paper
    https://www.math.nyu.edu/faculty/avellane/Litterman1991.pdf
    
    """
    def __init__(self, **params):
        return

    def create_data(self, csv_file="external/MVA2-PCA/Marktzinsen_mod.csv"):
        df = pd.read_csv(csv_file, sep=',')

        df['Datum'] = pd.to_datetime(df['Datum'],infer_datetime_format=True)
        
        df.set_index('Datum', drop=True, inplace=True)
        
        df.index.names = [None]
        
        df.drop('Index', axis=1, inplace=True)
        
        dt = df.transpose()

        return df

    def plot_YC(self, df):
        plt.figure(figsize=(12,6))

        plt.plot(df.index, df)
        plt.xlim(df.index.min(), df.index.max())
        # plt.ylim(0, 0.1)
        plt.axhline(y=0,c="grey",linewidth=0.5,zorder=0)
        for i in range(df.index.min().year, df.index.max().year+1):
            plt.axvline(x=df.index[df.index.searchsorted(pd.datetime(i,1,1))-1],
                        c="grey", linewidth=0.5, zorder=0)


    def doPCA(self, df, doDiff=True):
        """
        Parameters
        ----------
        doDiff: Boolean.  Take first order difference of data before performing PCA
        """
        # calculate the PCA (Eigenvectors & Eigenvalues of the covariance matrix)
        pcaA = PCA(n_components=3, copy=True, whiten=False)

        # pcaA = KernelPCA(n_components=3,
        #                  kernel='rbf',
        #                  gamma=2.0, # default 1/n_features
        #                  kernel_params=None,
        #                  fit_inverse_transform=False,
        #                  eigen_solver='auto',
        #                  tol=0,
        #                  max_iter=None)

        # transform the dataset onto the first two eigenvectors
        # kjp: change to diff
        df_in = df.copy()

        if doDiff:
            df_in = df_in.diff(axis=0).dropna()


        pcaA.fit(df_in)
        dpca = pd.DataFrame(pcaA.transform(df_in))
        dpca.index = df_in.index


        return pcaA, dpca

    def plot_cum_variance(self, pca):
        pch = PCA_Helper()
        return pch.plot_cum_variance(pca)

    def plot_components(self, pcaA, xlabel="Original feature #" , ylabel="Original feature value"):
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        ax.set_title('First {0} PCA components'.format(np.shape(np.transpose(pcaA.components_))[-1]))

        ax.plot(np.transpose(pcaA.components_) )

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.legend(["PC 1", "PC 2", "PC 3"])
