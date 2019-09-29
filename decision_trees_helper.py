import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pdb

import os
import subprocess

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin

# Tools
from sklearn import preprocessing, model_selection 
from sklearn.tree import export_graphviz

# Models
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

class SexToInt(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        I am really cheating here ! Am ignoring all columns except for "Sex"
        """
        
        # To see that I am cheating, look at the number of columns of X !
        print("SexToInt:transform: Cheating alert!, X has {c} columns.".format(c=X.shape[-1]) )
        
        sex = X["Sex"]
        X["Sex"] = 0
        X[ sex == "female" ] = 1
        
        return(X)

class TitanicHelper():
    def __init__(self, **params):
        TITANIC_PATH = os.path.join("./external/jack-dies", "data")

        train_data = pd.read_csv( os.path.join(TITANIC_PATH, "train.csv") )
        test_data  = pd.read_csv( os.path.join(TITANIC_PATH, "test.csv")  )

        target_name = "Survived"

        self.train_data = train_data
        self.test_data  = test_data
        self.target_name = target_name

        return

    def make_numeric_pipeline(self, features):
        num_pipeline = Pipeline([
                ("select_numeric", DataFrameSelector( features )),
                ("imputer", SimpleImputer(strategy="median")),
                ])
        
        return num_pipeline

    def make_cat_pipeline(self, features):
        cat_pipeline = Pipeline([
                ("select_cat", DataFrameSelector( features )),
                ("imputer", MostFrequentImputer()),
                ("sex_encoder", SexToInt() ),
                ])

        return cat_pipeline

    def make_pipeline(self, num_features=["Age", "SibSp", "Parch", "Fare"], cat_features=["Sex", "Pclass"] ):
        num_pipeline = self.make_numeric_pipeline(num_features)
        cat_pipeline = self.make_cat_pipeline(cat_features)

        preprocess_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", num_pipeline),
                ("cat_pipeline", cat_pipeline),
                ]
                                           )
        feature_names = num_features.copy()
        feature_names.extend(cat_features)

        return preprocess_pipeline, feature_names

    def run_pipeline(self, pipeline, data):
        # Run the pipelinem return an ndarray
        data_trans = pipeline.fit_transform(data)

        return data_trans


    def make_logit_clf(self):
        # New version of sklearn will give a warning if you don't specify a solver (b/c the default solver -- liblinear -- will be replaced in future)
        logistic_clf = linear_model.LogisticRegression(solver='liblinear')

        return logistic_clf

    def make_tree_clf(self, max_depth):
        tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        return tree_clf

    def fit(self, clf, train_data, target_name):
        pipeline, feature_names = self.make_pipeline()
        self.feature_names = feature_names

        # n.b., target_name MUST NOT be part of the transformation pipeline
        #   This is to ensure that it is dropped from train_transf
        train_transf = self.run_pipeline(pipeline, train_data)

        train_transf_df = pd.DataFrame(train_transf, columns=feature_names)

        y_train = train_data[target_name]
        X_train = train_transf_df

        self.X_train = X_train
        self.y_train = y_train

        clf.fit(X_train, y_train)
    
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        self.scores = scores

        return clf

    def export_tree(self, tree_clf, out_file, feature_names, target_classes, to_png=True):
        dot_file = out_file + ".dot"

        ret = { "dot_file": dot_file }

        export_graphviz(
            tree_clf,
            out_file=dot_file,
            feature_names=feature_names,
            class_names=target_classes,
            rounded=True,
            filled=True
            )

        if to_png:
            png_file = out_file + ".png"
            cmd = "dot -Tpng {dotf} -o {pngf}".format(dotf=dot_file, pngf=png_file)
            ret["png_file"] = png_file

            retval = subprocess.call(cmd, shell=True)
            ret["dot cmd rc"] = retval

        return ret

    def make_titanic_png(self, max_depth=2):
        train_data = self.train_data
        target_name = self.target_name

        tree_clf = self.make_tree_clf(max_depth=max_depth)
        self.fit(tree_clf, train_data, target_name)

        fname = "images/titanic_{depth:d}level".format(depth=max_depth)
        self.export_tree(tree_clf, fname, self.feature_names, [ "No", "Yes"] )

        return fname

    def partition(self, X, y, conds=[]):
        mask = pd.Series(data= [ True ] * X.shape[0], index=X.index )
        X_filt = X.copy()

        for cond in conds:
            (col, thresh) = cond
            print("Filtering column {c} on {t}".format(c=col, t=thresh) )
            cmp = X[ col ] <= thresh
            mask = mask & cmp

        return (X[mask], y[mask], X[~mask], y[~mask])
            

class GiniHelper():
    def __init(self, **params):
        return

    def plot_Gini(self):
        """
        Plot Gini score of binary class
        """
        p     = np.linspace(0,1, 1000)
        not_p = 1 - p
        
        gini = 1 - np.sum(np.c_[p, not_p]**2, axis=1)
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        _ = ax.plot(p, gini)
        _ = ax.set_xlabel("p")
        _ = ax.set_ylabel("Gini")
        _ = ax.set_title("Gini: Two class example")

        return fig, ax

    def make_logic_fn(self, num_features=3, target_name="target"):
        rows = []
        fstring = "{{:0{:d}b}}".format(num_features)

        for i in range(0, 2**num_features):
            row =[ int(x) for x in list( fstring.format(i) ) ]
            rows.append(row)

        feature_names = [ "feat {i:d}".format(i=i)  for i in range(1,num_features+1) ]
        df = pd.DataFrame.from_records(rows, columns=feature_names)

        target =  ( (df["feat 1"] == 1) | df["feat 2"] == 1 ) & (df["feat 3"] == 1)

        df[target_name] = target.astype(int)

        return df, target_name, feature_names
        
    def make_logic_dtree(self, df, target_name):
        tree_clf = DecisionTreeClassifier( random_state=42 )
        
        y = df[target_name]
        X = df.drop(columns=[target_name])
        tree_clf.fit(X, y)

        return tree_clf

    def make_logicTree_png(self):
       df, target_name, feature_names = self.make_logic_fn()

       self.df_lt, self.target_name_lt, self.feature_names_lt = df, target_name, feature_names

       th = TitanicHelper()
       tree_clf = self.make_logic_dtree(df, target_name)

       fname = "images/logic_tree"
       th.export_tree(tree_clf, fname, feature_names, [ "No", "Yes"] )

       return fname

    def gini(self, df, target_name, feature_names, noisy=False):
        
        count_by_target = df[ target_name ].value_counts()
        count_total = count_by_target.values.sum()

        # Compute frequencies
        freq = count_by_target/count_total

        # Square the frequencies
        freq2 = freq ** 2
        
        # Compute Gini
        gini = 1 - freq2.sum()

        if noisy:
            print("Gini, by hand:")
            print("Count by target:\n\t")
            print(count_by_target)
            print("Frequency by target:\n\t")
            print(freq)
            print ("\n1 - sum(freq**2) = {g:0.3f}".format(g =gini) )

        return gini

    def cost(self, df, target_name, feature_names, noisy=False):
        for feature_name in feature_names:
            count_by_feature_value = df[feature_name].value_counts()
            count_total = count_by_feature_value.values.sum()

            feature_values = count_by_feature_value.index.tolist()
            
            # Eliminate the max value since <= max includes everything
            feature_values = sorted(feature_values)[:-1]

            for feature_value in feature_values:
                cond = df[feature_name] <= feature_value
                df_left = df[ cond ]
                df_right = df[ ~ cond ]

                gini_left  = self.gini(df_left, target_name, feature_names)
                gini_right = self.gini(df_right, target_name, feature_names)
                
                count_left  = df_left.shape[0]
                count_right = df_right.shape[0]
                

                cost = (count_left/count_total) * gini_left + (count_right/count_total) * gini_right

                if noisy:
                    print("Split feature {f:s} on {fv:0.2f}".format(f=feature_name, fv=feature_value))
                    print("\tG_left (# = {lc:d}) = {gl:0.3f}, G_right (# = {rc:d}) = {gr:0.3f}".format(gl=gini_left, gr=gini_right, lc=count_left, rc=count_right) )
                    print("\tweighted (G_left, G_right) = {c:0.3f}".format(c=cost) )
        return cost

class RegressionHelper():
    def __init__(self, **params):
        return

    def make_plot(self, seed=42):
        """
        Based on https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
        """
        # Create a random dataset
        rng = np.random.RandomState(seed)

        X = np.sort(5 * rng.rand(80, 1), axis=0)
        y = np.sin(X).ravel()
        y[::5] += 3 * (0.5 - rng.rand(16))

        # Predict
        X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

        th = TitanicHelper()

        # Fit and  Plot the results
        fig, ax = plt.subplots(1,2, figsize=(12,5))

        for i, depth in enumerate([2,5]):
            regr = DecisionTreeRegressor(max_depth=depth)
            regr.fit(X, y)

            y_1 = regr.predict(X_test)

            ax[i].scatter(X, y, s=20, edgecolor="black",
                          c="darkorange", label="data")

            ax[i].plot(X_test, y_1, color="cornflowerblue",
                 label="max_depth={d:d}".format(d=depth), linewidth=2)

            ax[i].set_xlabel("data")
            ax[i].set_ylabel("target")
            ax[i].set_title("Decision Tree Regression, max depth={d:d}".format(d=depth) )

            # Create the png
            fname = "images/tree_regress_depth_{d:d}".format(d=depth)
            th.export_tree(regr, fname, [ "X" ], [ "No", "Yes"] )

        return fig, ax
