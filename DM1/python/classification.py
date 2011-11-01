import numpy as np


def linear_regression(X, Y):
    """
    Fit linear regression model on X, Y
    """
    # Linear Regression
    X_2 = np.ones((X.shape[0], X.shape[1] + 1))
    X_2[:, :2] = X
    X = X_2

    a = np.linalg.inv(np.dot(X.T, X))
    b = np.dot(X.T, Y)
    beta = np.dot(a, b)

    return beta[:2], beta[2]


def logistic_regression(X, Y, max_iter=500, ridge=1e-10, verbose=True):
    """
    Compute logisitic regression
    """
    X_2 = np.ones((X.shape[0], X.shape[1] + 1))
    X_2[:, :2] = X
    X = X_2
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
    beta = theta[:2]
    u = theta[2]
    return beta, u


def logistic_regression_predict(X, theta, gamma):
    """
    Predict the label for X, depending on beta and gamma
    """
    p = 1. / (1 + np.exp(-np.dot(X, theta) - gamma))
    Y = p > 0.5
    return Y.astype(int)


def linear_regression_predict(X, theta, gamma):
    """
    Predict the label for X with a linear regression model
    """

    p = np.dot(X, theta) + gamma
    Y = p > 0.5
    return Y.astype(int)


def LDA(X, Y):
    n = Y.shape[0]
    p = Y.sum() / n
    m_1 = (Y * X).sum(axis=0) * 1 / (Y.sum())
    m_0 = ((1 - Y) * X).sum(axis=0) * 1 / ((1 - Y).sum())


    a = np.dot((X -m_1).T,(Y * (X - m_1)))
    b = np.dot((X - m_0).T, ((1 - Y) * (X - m_0)))
    S = 1. / n * (a + b)

    S_inv = np.linalg.inv(S)

    beta = np.dot(S_inv, (m_1 - m_0))

    c = np.dot(np.dot((m_1).T, S_inv), m_1) - np.dot(np.dot((m_0).T, S_inv),
    m_0)
    d = np.log(p / (1 - p))
    gamma = - 1. / 2 * c - d
    return beta, gamma


def QDA(X, Y):
    n = Y.shape[0]
    p = Y.sum() / n
    m_1 = (Y * X).sum(axis=0) * 1 / (Y.sum())
    m_0 = ((1 - Y) * X).sum(axis=0) * 1 / ((1 - Y).sum())

    S_1 = (Y * (X - m_1)).sum(axis=0) / (Y.sum())
    S_0 = ((1 - Y) * (X - m_0)).sum(axis=0) / ((1 - Y).sum())

    return p, m_1, m_0, S_1, S_0


def error(Yt, Y):
    Yt.shape = Y.shape
    error = Y != Yt
    return error.astype(float).sum() / Y.shape[0]


