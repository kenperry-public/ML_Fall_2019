import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split

import functools
import itertools

from ipywidgets import interact, fixed

import pdb

class Classification_Helper():
    def __init__(self, **params):
        return


from sklearn.datasets import make_classification
X,y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                    n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                          class_sep=1.0,
                          random_state=10
                   )
plt.scatter(X[:,0], X[:,1], c=y)
