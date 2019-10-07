import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_circles
from sklearn.svm import SVC

import functools

from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed

class SVM_Helper():
    def __init__(self, **params):
        self.X, self.y = None, None
        return

    def sigmoid(self, x):
        x = 1/(1+np.exp(-x))
        return x


    def plot_pos_examples(self, score, ax, hinge_pt=None):
        # Apply sigmoid to turn into probability
        p = self.sigmoid(score)

        neg_logs =  - np.log(p)
        _= ax.plot(score, neg_logs, label="- log p")
        _= ax.set_title("Positive examples")
        _= ax.set_xlabel("Score")

        if hinge_pt is not None:
            # Hinge at intercept interc, slope 0.5
            hinge = np.zeros( p.shape[0] )
            interc = hinge_pt
            hinge[ score <= interc ] = 0.5 * ( - ( score[ score <= interc]) + interc)
            _= ax.plot(score, hinge, label="hinge")
            _= ax.legend()


    def plot_neg_examples(self, score, ax, hinge_pt=None):
        # Apply sigmoid to turn into probability
        p = self.sigmoid(score)

        neg_logs =  -np.log(1-p)

        _= ax.plot(score, neg_logs, label="- log(1-p)")
        _= ax.set_title("Negative examples")
        _= ax.set_xlabel("Score")

        if hinge_pt is not None:
            # Hinge at intercept interc, slope 0.5
            hinge = np.zeros( p.shape[0] )
            interc = - hinge_pt
            hinge[ score >= interc ] = 0.5 * ( ( score[ score >= interc]) - interc )
            _= ax.plot(score, hinge, label="hinge")

        _= ax.legend()

    def plot_log_p(self, hinge_pt=None):
        fig, axs = plt.subplots(1,2, figsize=(12, 4.5))

        score = np.linspace(-3,+3, num=100)
        _ = self.plot_pos_examples(score, axs[0], hinge_pt=hinge_pt)
        _ = self.plot_neg_examples(score, axs[1], hinge_pt=hinge_pt)
        
        fig.tight_layout()

    def plot_hinges(self, hinge_pt=0):
        fig, axs = plt.subplots(1,2, figsize=(12, 4.5))
        score = np.linspace(-3,+3, num=100)
        hinge_p = np.maximum(hinge_pt, -score)
        hinge_n = np.maximum(hinge_pt,  score)

        _= axs[0].plot(score, hinge_p)
        _= axs[0].set_label("Score")
        _= axs[0].set_title("Positive examples")

        _= axs[1].plot(score, hinge_n)
        _= axs[1].set_label("Score")
        _= axs[1].set_title("Negative examples")

        fig.tight_layout()


    # Adapted from external/PythonDataScienceHandbook/notebooks/05.07-Support-Vector-Machines.ipynb
    
    def make_circles(self, plot=False):
        X, y = make_circles(100, factor=.1, noise=.1)

        if plot:
            plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            
        return X,y

    def plot_svc_decision_function(self, model, ax=None, plot_support=True):
        """
        Plot the decision function for a 2D SVC
        """
        
        if ax is None:
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        # create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)

        # plot decision boundary and margins
        ax.contour(X, Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        # plot support vectors
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                       model.support_vectors_[:, 1],
                       s=300, linewidth=1, facecolors='none');
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
            

    def circles_linear(self, X, y):
        clf = SVC(kernel='linear').fit(X, y)

        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        self.plot_svc_decision_function(clf, plot_support=False);

  
    def plot_3D(self, elev=30, azim=30, X=[], y=[]):
        ax = plt.subplot(projection='3d')
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y, s=50, cmap='autumn')
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel('r')

        return ax

    def circles_radius_transform(self, X):
        r = -(X ** 2).sum(1)

        X_new = np.concatenate((X,r[:, np.newaxis]), axis=1)
        return X_new

    def circles_rbf_transform(self, X):
        r = np.exp( -(X ** 2).sum(1) )

        X_new = np.concatenate((X,r[:, np.newaxis]), axis=1)
        return X_new

    def circles_square_transform(self, X):
        r = np.zeros( X.shape[0] )
        r[ np.all(np.abs(X) <= 0.5, axis=1) ] = 1

        X_new = np.concatenate((X,r[:, np.newaxis]), axis=1)
        return X_new

