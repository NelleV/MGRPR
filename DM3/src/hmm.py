import numpy as np

from utils import load_data

data = load_data('EMGaussienne.data')
data = data[:100]
K = 4
a = 0.25 * np.ones((K, K))
a[0] = 0.21
a[2] = 0.29

T = 100


def _calculate_normal(X, mu, sigma):
    """
    Calculate the probability p of X following normal law
    """
    a = 1. / (np.sqrt(2 * np.pi) * np.linalg.det(sigma))
    S_inv = np.linalg.inv(sigma)
    b = np.dot(np.dot((X - mu), S_inv), (X - mu).T)
    p = a * np.exp(-1. / 2 * b)
    return p


pi = 0.5  # Valeur bidon
mu = [np.array([0, 1]),
      np.array([1, 1]),
      np.array([1, 0]),
      np.array([0, 2])]

sigma = [0.6 * np.identity(2),
         np.identity(2),
         np.identity(2),
         np.identity(2)]


def e_step(data, a, pi, mu, sigma):
    """
    """
    q = np.zeros((len(data), 1))
    alpha = np.zeros((len(q), K))
    alpha_norm = np.zeros((len(q), 1))
    beta = np.zeros((len(q), K))

    for i, element in enumerate(q):
        p = np.array([_calculate_normal(data[i, :], mu[j], sigma[j])
                        for j in range(K)])

        if i == 0:
            # Let's initialize the chain
            alpha[i, :] = p * pi
            alpha_norm[i] = alpha[i, :].sum()
            alpha[i, :] /= alpha_norm[i]
        else:
            for k in range(K):
                alpha[i, k] = np.dot(alpha[i - 1, :] * a[:, k], p)
            alpha_norm[i] = alpha[i, :].sum()
            alpha[i, k] /= alpha_norm[i]

    for i, element in enumerate(q):
        p = np.array([_calculate_normal(data[len(q) - 1 - i, :],
                                        mu[j],
                                        sigma[j])
                        for j in range(K)])

        if i == 0:
            beta[len(q) - 1 - i, :] = 0.25, 0.25, 0.25, 0.25
            #beta[len(q) - 1 - i, :] /= alpha_norm[len(q) - 1 - i]
        else:
            for k in range(K):
                beta[len(q) - 1 - i, k] = np.dot(beta[len(q) - i] * a[:, k], p)
            #beta[len(q) - 1 - i, :] *= alpha_norm[len(q) - 1 - i]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1).reshape((len(gamma), 1))[-1]

    xi = np.zeros((len(q) - 1, K, K))
    for i, element in enumerate(q[:-1]):
        p = np.array([_calculate_normal(data[i + 1, :], mu[j], sigma[j])
                    for j in range(K)])
        for k in range(K):
            for g in range(K):
                xi[i, k, g] = alpha[i, k] * gamma[i + 1, g] * p[g] * a[k, g]
                xi[i, k, g] /= gamma[i + 1, g]
    return alpha, beta, gamma, xi


# Estimation des parametres !
def m_step(gamma, xi, data, sigma, mu, a, pi):
    pi = gamma[0]
    for i in range(K):
        for j in range(K):
            a[i, j] = xi[:, i, j].sum() / gamma[:, i].sum()

        for indx, element in enumerate(data):
            b = (element - mu[i]).reshape((len(element), 1))
            sigma[i] += gamma[indx, i] * np.dot(b,
                                                b.T)
        sigma[i] /= gamma[:, i].sum()

        mu[i] = (gamma[:, i].reshape((len(gamma), 1)) * data).sum(axis=0)
        mu[i] /= gamma[:, i].sum()
    return pi, a, gamma, mu, sigma

max_iter = 2
alpha, beta, gamma, xi = e_step(data, a, pi, mu, sigma)
pi, a, gamma, mu, sigma = m_step(gamma, xi, data, sigma, mu, a, pi)
