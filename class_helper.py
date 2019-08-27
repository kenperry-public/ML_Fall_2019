import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split

import functools

from ipywidgets import interact, fixed

import pdb

class Classification_Helper():
    def __init__(self, **params):
        return

    def load_digits(self):
        digits = datasets.load_digits()
        X_digits = digits.data / digits.data.max()
        y_digits = digits.target

        self.size = int( np.sqrt( X_digits.shape[1] )  )
        return X_digits, y_digits

    def split_digits(self, X, y, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=100,
            shuffle=True, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def plot_digits(self, X_digits, y_digits, digits_per_row=10):
        digits = range(y_digits.min(), y_digits.max() +1)

        (num_rows, num_cols) = (len(digits) , digits_per_row)
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12,8))

        num_shown = 0

        for row, digit in enumerate(digits):
            this_digits = X_digits[ y_digits == digit ]
            imgs = [ img.reshape(self.size, self.size) for img in this_digits[:num_cols, :] ]
            for col, img in enumerate(imgs):
                ax = axs[row,col]
                _ = ax.set_axis_off()

                _ = ax.imshow(img, cmap = mpl.cm.binary)

        return fig, axs
    
    def fit_digits(self, X_digits, y_digits):
        X_train, X_test, y_train, y_test = self.split_digits(X_digits, y_digits)
        knn = neighbors.KNeighborsClassifier()
        logistic = linear_model.LogisticRegression(solver='lbfgs',
                                                   max_iter=1000,
                                                   multi_class='multinomial')

        print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
        print('LogisticRegression score: %f'
              % logistic.fit(X_train, y_train).score(X_test, y_test))

        models = { "knn": knn,
                   "lr" : logistic
                   }

        return  X_train, X_test, y_train, y_test, models

    def predict_digits(self, model, X_digits, y_digits):
        preds = model.predict(X_digits)

        digits_per_row = 5
        num_rows = X_digits.shape[0] // digits_per_row
        num_rows += num_rows + 1 if (0 != X_digits.shape[0] % digits_per_row) else 0
        num_cols = digits_per_row
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, num_rows *1.5))

        plt.subplots_adjust(hspace=0.32)
        
        for i in range( preds.shape[0]):
            img, pred, target = X_digits[i], preds[i], y_digits[i]
            row, col = i // num_cols, i % num_cols
            ax = axs[row,col]
        
            _= ax.imshow( img.reshape(self.size, self.size), cmap = mpl.cm.binary)
            _ = ax.set_axis_off()
            
            if pred == target:
                label = "Correct {dig:d}".format(dig=pred)
            else:
                label = "Incorrect: Predict {p:d}, is {t:d}".format(p=pred, t=target)
            ax.set_title(label)

        return fig, axs
