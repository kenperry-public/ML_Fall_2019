import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split

import functools

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import random 
from sklearn import linear_model


import pdb

class Transformation_Helper():
    def __init__(self, **params):
        return

    def gen_data(self, m=30, mean=30, std=10, slope=5, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
       
        length = mean + std*np.random.randn(m) 
        width  = mean + std*np.random.randn(m) 

        area = length * width

        # Clip data
        area = np.where( area < 350, 350, area)
        area = np.where( area > 2200, 2200, area)

        noise =   ( std**2  * np.random.randn(m) ) * slope/2
        price = noise + slope * area

        df = pd.DataFrame( { "length": length,
                             "width":  width,
                             "area":   area,
                             "price":  price
                             }
                           )

        df = df.sort_values(by=["area"])
                            
        return df

class InfluentialPoints_Helper():
    def __init__(self, **params):
        # Random number generator
        rng = np.random.RandomState(42)
        self.rng = rng
        
        return

    def fit(self, x,y):
        # Fit the model to x, y
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        
        y_pred = regr.predict(x)
        return (x, y_pred, regr.coef_[0])

    def gen_data(self,num=10):
        rng = self.rng
        
        x = np.linspace(-10, 10, num).reshape(-1,1)
        y =  x + rng.normal(size=num).reshape(-1,1)

        self.x, self.y = x, y
        
        return(x,y)

    
    def plot_update(self, x,y, fitted):
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        plt.ion()

        # Scatter plot of data point
        _ = ax.scatter(x, y, color='black', label="true")

        # Line of best fit
        ax.plot( fitted[0], fitted[1], color='red', label='fit')

        slope = fitted[2]

        ax.set_title("Slope = {s:.2f}".format(s=slope[0]))
        # count += 1
        # if count % max_count == 0:
        fig.canvas.draw()


    def fit_update(self, x_l, y_l):
        # Retrieve original data
        x, y, fitted = self.x, self.y, self.fitted
        
        print("called with ", x_l, y_l)
        # Update a single y_value
        # - the element of array y at index x_l will be changed to y_l
        x_update, y_update = x.copy(), y.copy()
        y_update[ x_l ] = y_l

        # Fit the model to the updated data
        fitted = self.fit(x_update,y_update)

        # Plot the points and the fit, on the updated data
        self.plot_update(x_update, y_update, fitted)

    def create_plot_updater(self):
        # Return a plot update function
        # NOTES:
        # - functools.partial does not return an object of the necessary type (bound method_
        # - See:
        # --- https://stackoverflow.com/questions/16626789/functools-partial-on-class-method
        # --- https://stackoverflow.com/questions/48337392/how-to-bind-partial-function-as-method-on-python-class-instances
        # --- so this is our own version of partial
        # - the "interact" method has *named* arguments (e.g., x_l, y_l)
        # -- the function we create must use the *same* names for its formal parameters
        def fn(x_l, y_l):
            return self.fit_update(x_l,y_l)
    
        return fn

    def plot_init(self):
        x, y = self.x, self.y

        # Fit a line to the x,y data
        fitted = self.fit(x,y)

        # Save the fitted data
        self.fitted = fitted

        # Create  function to update the fit and the plot
        plot_updater = self.create_plot_updater()
        return plot_updater
    

    def plot_interact(self, fit_update_fn):
        x, y = self.x, self.y
        num = len(x)
        
        interact(fit_update_fn,
                 x_l=widgets.IntSlider(min=0,  max=num-1,  step=1,
                                       value=int(num/2),
                                       continous_update=False),
                 y_l=widgets.IntSlider(min=y.min(), max=y.max(), step=1, 
                                       value=y[ int(num/2)], 
                                       continous_update=False)
                 )

class ShiftedPrice_Helper():
    def __init__(self, **params):
        rng = np.random.RandomState(42)
        self.rng = rng
        
        return

    def gen_data(self):
        num_series = 3
        th = Transformation_Helper()
        
        dfs = [ th.gen_data(random_seed=42+i) for i in range(0,4) ]

        return dfs

    def plot_data(self, dfs):
        _= plt.scatter(dfs[0]["area"],         dfs[0]["price"], label="$time_0$")
        _= plt.scatter(dfs[1]["area"],  2000 + dfs[1]["price"], label="$time_1$")
        _= plt.legend()

class RelativePrice_Helper():
    def __init__(self, **params):
        # Random number generator
        rng = np.random.RandomState(42)
        self.rng = rng
        
        return

    def relative_price(self):
        return (1, 1, 2,3)
    
    def gen_data(self):
        # Generate a set of random, near linear data (number of series: num_series)
        num_series = 3
        th = Transformation_Helper()
        dfs = [ th.gen_data(random_seed=42+i) for i in range(0,num_series+1) ]

        # Inflate each series by a different amount (rel_price)
        rel_price = self.relative_price()

        # dfs_i re the inflates series
        dfs_i = []
        for i in range(0, len(dfs)):
            df = dfs[i].copy()
            df["price"] *= rel_price[i]
            dfs_i.append(df)

        
        return dfs_i

    def plot_data(self, dfs):
        _= plt.scatter(dfs[0]["area"],  dfs[0]["price"], label="$time_0$")
        _= plt.scatter(dfs[2]["area"],  dfs[2]["price"], label="$time_1$")
        _= plt.scatter(dfs[3]["area"],  dfs[3]["price"], label="$time_2$")
        _= plt.legend()
