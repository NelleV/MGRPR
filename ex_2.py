#
# IRLS
#
# FIXME constante missing !!!!

import numpy as np
import pylab as pl

from classification import logistic_regression
from utils import load_data

verbose = True
max_iter = 500

X, Y = load_data('classificationA.train')
W = np.ones((X.shape[0], 1))
ridge = 1e-10
theta = np.zeros((X.shape[1], 1))

for iter in range(max_iter):
    print iter
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

beta, u = logistic_regression(X, Y)
# Calculate the line p(y = 1|x) = 0.5


# Plot
fig = pl.figure(1)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
my_members = Y == 0
my_members.shape = (my_members.shape[0])
ax = fig.add_subplot(1, 1, 1)
ax.plot(X[my_members, 0], X[my_members, 1],
        'w', markerfacecolor=colors[0], marker = '.')

my_members = Y == 1
my_members.shape = (my_members.shape[0])
ax.plot(X[my_members, 0], X[my_members, 1],
        'w', markerfacecolor=colors[1], marker = '.')

x_beta = [[i] for i in np.linspace(X.min(), X.max(), 100)]
y_beta =  (0.5 - u - beta[0] * np.linspace(X.min(), X.max(), 100)) * 1 / beta[1]
ax.plot(x_beta, y_beta, color=colors[2], linewidth=1)
pl.show()



