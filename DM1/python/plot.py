# -*- coding: utf-8 -*-
import numpy as np

import classification
from utils import load_data

from matplotlib import pyplot as plt

# Calculate the errors for the three datasets

XA, YA = load_data('classificationA.train')
XB, YB = load_data('classificationB.train')
XC, YC = load_data('classificationC.train')
XtA, YtA = load_data('classificationA.test')
XtB, YtB = load_data('classificationB.test')
XtC, YtC = load_data('classificationC.test')

# Let's plot the data

def plot(X, Y, XtA, title="ClassificationA.png"):
    fig = plt.figure()
    colors = ['#4EACC5', '#FF9C34', '#aaaaaa', '#4E9A06', '#00465F', "#7E2007"]
    my_members = Y == 0
    my_members.shape = (my_members.shape[0])
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(X[my_members, 0], X[my_members, 1],
            'w', markerfacecolor=colors[0], marker = '.')

    my_members = Y == 1
    my_members.shape = (my_members.shape[0])
    ax.plot(X[my_members, 0], X[my_members, 1],
            'w', markerfacecolor=colors[1], marker = '.')


    beta, u = classification.LDA(X, Y)
    YtcA = classification.logistic_regression_predict(XtA, beta, u)
    x_beta = [[i] for i in np.linspace(X.min(), X.max(), 100)]
    y_beta =  (- u - beta[0] * np.linspace(X.min(), X.max(), 100)) * 1 / beta[1]
    ax.plot(x_beta, y_beta, color=colors[3], linewidth=1)


    beta, u = classification.logistic_regression(X, Y, verbose=False)
    x_beta = [[i] for i in np.linspace(X.min(), X.max(), 100)]
    y_beta =  (- u - beta[0] * np.linspace(X.min(), X.max(), 100)) * 1 / beta[1]
    ax.plot(x_beta, y_beta, color=colors[4], linewidth=1)

    YtcA = classification.logistic_regression_predict(XtA, beta, u)

    beta, u = classification.linear_regression(X, Y)
    YtcA = classification.linear_regression_predict(XtA, beta, u)
    x_beta = [[i] for i in np.linspace(X.min(), X.max(), 100)]
    y_beta =  (0.5 - u - beta[0] * np.linspace(X.min(), X.max(), 100)) * 1 / beta[1]
    ax.plot(x_beta, y_beta, color=colors[5], linewidth=1)

    labels = ('unknown', 'label 0', 'label 1', 'LDA model', 'logistic regression', 'linear regression')
    legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='small')

    plt.show()
    plt.savefig(title)
    


plot(XA, YA, XtA, title="classificationA.png")
plot(XB, YB, XtB, title="classificationB.png")
plot(XC, YC, XtC, title="classificationC.png")

