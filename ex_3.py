# Linear Regression

import numpy as np

from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt


from classification import linear_regression
from utils import load_data

verbose = True
max_iter = 500

X, Y = load_data('classificationA.train')

beta, u = linear_regression(X, Y)

# Let's plot the result
fig = plt.figure(1)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
my_members = Y == 0
my_members.shape = (my_members.shape[0])
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

for direction in ["xzero", "yzero"]:
    ax.axis[direction].set_axisline_style("-|>")
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    ax.axis[direction].set_visible(False)


ax.plot(X[my_members, 0], X[my_members, 1],
        'w', markerfacecolor=colors[0], marker = '.')

my_members = Y == 1
my_members.shape = (my_members.shape[0])
ax.plot(X[my_members, 0], X[my_members, 1],
        'w', markerfacecolor=colors[1], marker = '.')

x_beta = [[i] for i in np.linspace(X.min(), X.max(), 100)]
#y_beta = 1 / beta[0] + np.linspace(x.min(), x.max(), 100) * 1 / beta[1]
y_beta =  (0.5 - u - beta[0] * np.linspace(X.min(), X.max(), 100)) * 1 / beta[1]

ax.plot(x_beta, y_beta, color=colors[2], linewidth=1)

plt.show()


