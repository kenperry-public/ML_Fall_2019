def plot_fit(X, y, ax=ax, on_idx=0):
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
    sort_idx = X[:, on_idx].argsort()
    X_sorted = X[ sort_idx,:]
    y_sorted = y[ sort_idx,:]
    
    _ = ax.plot(X_sorted[:, on_idx] , y_sorted, color="red")
    
def transform(X):
    """
    Add a column to X with squared values
    
    Parameters
    ----------
    X: ndarray of features
    """
    X_p2 = np.concatenate( [X, X **2], axis=1)
    return X_p2
    
def run_regress(X,y, run_transforms=False):
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
    X_train, X_test, y_train, y_test = split(X,y)
    
    # Transform X's
    if (run_transforms):
        X_train = transform(X_train)
        X_test  = transform(X_test)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    _ = regr.fit(X_train, y_train)

    # The coefficients
    print('Coefficients: \n', regr.intercept_, regr.coef_)
    # Lots of predictions: predict on entire test set
    y_pred = regr.predict(X_test)

    # Explained variance score: 1 is perfect prediction
    print("R-squared (test): {:.2f}".format(r2_score(y_test, y_pred)) )

    y_pred_train = regr.predict(X_train)
    print("R-squared (train): {:.2f}".format(r2_score(y_train, y_pred_train)) )

    # Plot predicted ylabel (red) and true label (black)
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)

    _ = ax.scatter(X_test[:,0], y_test, color='black')
    _ = ax.scatter(X_test[:,0], y_pred, color="red")

    # _ = ax.plot(X_test[:,0], y_pred, color="red")
    plot_fit(X_test, y_pred, ax=ax, on_idx=0)
    _ = ax.set_xlabel(xlabel)
    _ = ax.set_ylabel(ylabel)
    
    return

def plot_resid(X, y, y_pred):
    resid_curve = y - y_pred
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    ax.scatter(X, resid_curve)
    _ = ax.set_xlabel(xlabel)
    _ = ax.set_ylabel("Residual")
