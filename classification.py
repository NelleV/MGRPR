import numpy as np

from sklearn.metrics.pairwise import euclidian_distance

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


def logistic_regression(X, Y, max_iter=500, ridge=1e-10, verbose=True):
    """
    Compute logisitic regression
    """
    W = np.ones((X.shape[0], 1))
    theta = np.zeros((X.shape[1], 1))

    for iter in range(max_iter):
        old = theta.copy()
        h = np.dot(X, theta)
        m = 1. / (1 + np.exp(-h))
        W = np.diag((m * (1 - m)).flatten())
        a = np.dot(X.T, (Y - m))
        b = np.linalg.inv(np.dot(X.T, np.dot(W, X)))
        theta = theta + np.dot(b, a)

        if ((old - theta)**2).sum() < ridge:
            if verbose:
                print "got out at iteration", iter
            break

    u = (Y - np.dot(X, theta)).mean()
    return theta, u


def error(Y, Yt):
    return (Y - Yt)**2.sum()


