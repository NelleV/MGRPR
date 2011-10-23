import numpy as np

def linear_regression(X, Y):
    """
    Fit linear regression model on X, Y
    """
    # Linear Regression
    a = np.linalg.inv(np.dot(X.T, X))
    b = np.dot(X.T, Y)
    beta = np.dot(a, b)

    u = (Y - np.dot(X, beta)).mean()
    return beta, u



