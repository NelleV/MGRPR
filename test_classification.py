import numpy as np

from matplotlib import pyplot as plt

from classification import linear_regression


def test_linear_regression():
    """
    Test set "stolen" from scikit learn
    """
    # this is our test set, it's just a straight line with some
    # gaussian noise
    xmin, xmax = -5, 5
    n_samples = 100
    X = np.array([[i] for i in np.linspace(xmin, xmax, n_samples)])
    Y = np.array(2 + 0.5 * np.linspace(xmin, xmax, n_samples) \
        + np.random.randn(n_samples, 1).ravel())

    beta, u = linear_regression(X, Y)

    plt.scatter(X, Y, color='black')
    plt.plot(X, np.dot(X, beta) + u, linewidth=1)
    plt.show()

