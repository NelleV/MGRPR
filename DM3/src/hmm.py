import numpy as np

from sklearn.mixture import GMM
from utils import load_data

data = load_data('EMGaussienne.data')
K = 4

gmm = GMM(n_components=K)
gmm.fit(data)

a = 0.25 * np.ones((K, K))
a[0] = 0.21
a[2] = 0.29

pi = 0.5  # Valeur bidon
mu = gmm.means.copy()
sigma = gmm.covars


def _calculate_normal(X, mu, sigma):
    """
    Calculate the probability p of X following normal law
    """
    a = 1. / (np.sqrt(2 * np.pi) ** len(sigma) * np.sqrt(np.linalg.det(sigma)))
    S_inv = np.linalg.inv(sigma)
    b = np.dot(np.dot((X - mu).T, S_inv), (X - mu))
    p = a * np.exp(-1. / 2 * b)
    return p


def compute_gamma(a, alpha):
    """
    """
    K = len(a)
    gamma = np.zeros((len(alpha), K))
    for i, element in enumerate(alpha):
        if i == 0:
            gamma[len(alpha) - 1 - i] = alpha[len(alpha) - 1 - i]
        else:
            for k in range(K):
                b = alpha[len(alpha) - i, k] * a[k, :]
                c = (alpha[len(alpha) - i] * a).sum(axis=1)
                gamma[len(alpha) - 1 - i, k] = np.dot(b / c,
                                                  gamma[len(alpha) - i])
    return gamma


def compute_alpha(data, pi, mu, sigma):
    """
    """
    alpha = np.zeros((len(data), K))
    alpha_norm = np.zeros((len(data), 1))

    for i, element in enumerate(data):
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
            alpha[i, :] /= alpha_norm[i]
    return alpha


def e_step(data, a, pi, mu, sigma):
    """
    """
    q = np.zeros((len(data), 1))
    beta = np.zeros((len(q), K))

    alpha = compute_alpha(data, pi, mu, sigma)
    gamma = compute_gamma(a, alpha)
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


max_iter = 1
for i in range(max_iter):
    alpha, beta, gamma, xi = e_step(data, a, pi, mu, sigma)
    pi, a, gamma, mu, sigma = m_step(gamma, xi, data, sigma, mu, a, pi)
q = gamma.argmax(axis=1)
