import time
import matplotlib.pyplot as plt
import numpy as np

import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

import pdb

class NN_Helper():
    def __init__(self, **params):
        self.X, self.y = None, None
        return

    def sigmoid(self, x):
        x = 1/(1+np.exp(-x))
        return x

    def sigmoid_grad(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2
    

    def plot_activations(self, x):
        sigm   = self.sigmoid(x)
        d_sigm = self.sigmoid_grad(x)
        d_tanh = 1 - np.tanh(x)**2
        d_relu = np.zeros_like(x) +  (x >= 0)

        fig, axs = plt.subplots(3,2, figsize=(16, 8))
        _ = axs[0,0].plot(x, sigm)
        _ = axs[0,0].set_title("sigmoid")
        
        _ = axs[0,1].plot(x, d_sigm)
        _ = axs[0,1].set_title("derivative sigmoid")
        
        _ = axs[1,0].plot(x, np.tanh(x))
        _ = axs[1,0].set_title("tanh")
        
        _ = axs[1,1].plot(x, d_tanh)
        _ = axs[1,1].set_title("derivative tanh")
        
        _ = axs[2,0].plot(x, np.maximum(0.0, x))
        _ = axs[2,0].set_title("ReLU")
        
        _ = axs[2,1].plot(x, d_relu)
        _ = axs[2,1].set_title("derivative ReLU")

        _ = fig.tight_layout()
        return fig, axs

    def NN(self, W,b):
        x = np.linspace(-100, 100, 100)
        z = W*x + b
        
        y = np.maximum(0, z)
        return { "x":x,
                 "y":y,
                 "W":W,
                 "b":b
                 }


    def plot_steps(self, xypairs):
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        for pair in xypairs:
            x, y, W, b = [ pair[l] for l in ["x", "y", "W", "b" ] ]
            _ = ax.plot(x, y, label="{w:d}x + {b:3.2f}".format(w=W, b=b))
            
            _ = ax.legend()
            _ = ax.set_xlabel("x")
            _ = ax.set_ylabel("activation")
            _ = ax.set_title("Step function creation")

        _ = fig.tight_layout()
        return fig, ax

        
            


    def plot_loss_fns(self):
        # prod = y * s(x)
        # Postive means correctly classified; negative means incorrectly classified
        prod  = np.linspace(-1, +2, 100)

        # Error if product is negative
        error_acc  =  prod < 0
        error_exp  =  np.exp( -prod )

        # Error is 0 when product is exactly 1 (i.e., s(x) = y = 1)
        error_sq    =  (prod -1 )** 2

        # Error is negative of product
        # Error unless product greater than margin of 1
        error_hinge =  (- (prod -1) ) * (prod -1 < 0)

        fig, ax = plt.subplots(1,1, figsize=(10,6))
        _ = ax.plot(prod, error_acc, label="accuracy")
        _ = ax.plot(prod, error_hinge, label="hinge")
        
        # Truncate the plot to keep y-axis small and comparable across traces
        _ = ax.plot(prod[ prod > -0.5], error_exp[ prod > -0.5], label="exponential")
        
        _ = ax.plot(prod[ prod > -0.5], error_sq[ prod > -0.5], label="square")
        _ = ax.legend()
        _ = ax.set_xlabel("error")
        _ = ax.set_ylabel("loss")
        _ = ax.set_title("Loss functions")
