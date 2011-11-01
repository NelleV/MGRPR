#
# IRLS
#

import numpy as np
import pylab as pl

from classification import logistic_regression
from utils import load_data

verbose = True
max_iter = 500
X, Y = load_data('classificationA.train')
beta, u = logistic_regression(X, Y)

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
y_beta =  (- u - beta[0] * np.linspace(X.min(), X.max(), 100)) * 1 / beta[1]
ax.plot(x_beta, y_beta, color=colors[2], linewidth=1)
pl.show()



