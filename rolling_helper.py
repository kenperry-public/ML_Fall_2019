import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split

import functools

from ipywidgets import interact, fixed
import random

import pdb

class Rolling_Helper():
    def __init__(self, **params):
        return


    def create_rolling(self, series, window=5, shift=1, drop_remainder=True):
        """
        Create rolling sub-series, each of length window, displaced by shift

        Parameters
        ----------
        series: pd.Series
        window: int.  Window size
        drop_remainder: Bool.  If True don't generate sequences less than full size, i.e., at end of series, there are not enough points

        Returns
        -------
        DataFrame df:
        - df.iloc[t] = series[t: t+window]

        """
        max_t = len(series) - window  if drop_remainder else len(series) -1

        rows = []
        
        t = 0

        # Roll forward thru the list in increments of size shift
        while (t < max_t):
            # Take a sub-series of length window; drop the index
            series_t = series[t:t+window].values

            rows.append(series_t)
            t += shift

        # Turn result into a DataFrame
        df = pd.DataFrame(rows)
        
        return df

    def apply_rolling(self, rows, func):
        """
        Apply a function to each row of a list

        Parameters
        ----------
        rows: List
        func: a function

        Returns
        -------
        List
        """
        result = []
        for t, row in enumerate(rows):
            row_res = func(row)
            result.append(row_res)

        return result

    def shuffle_rolling(self, df, random_seed=None):
        """
        Shuffle rows of dataframe

        Paramters
        ---------
        df: DataFrame

        Returns
        -------
        DataFrame
        """
        df_new = df.sample(frac=1.0, random_state=random_seed)

        return df_new

    def name_columns_rolling(self, df, format_string, func=lambda i: i):
        """
        Rename the columns of a DataFrame using a format string

        Paramters
        ---------
        df: DataFrame
        format_string: String, suitable for use in Python format statement
        - when format is called with this string, i will be bound to a row number of df
        
        """
        cols= []
        
        for i, col in enumerate(df.columns):
            col = format_string.format(i=func(i))
            cols.append(col)

        # Rename the columns
        df.columns = cols
        
        return df
        
    

    
                           

                           
        
