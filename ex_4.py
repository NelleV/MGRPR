# -*- coding: utf-8 -*-
import numpy as np

import classification
from utils import load_data
# Calculate the errors for the three datasets

XA, YA = load_data('classificationA.train')
XB, YB = load_data('classificationB.train')
XC, YC = load_data('classificationC.train')
XtA, YtA = load_data('classificationA.test')
XtB, YtB = load_data('classificationB.test')
XtC, YtC = load_data('classificationC.test')

# Jeu de données A

print "Jeu de données A"
print "****************"
print

beta, u = classification.LDA(XA, YA)
YtcA = classification.logistic_regression_predict(XtA, beta, u)
erreur = classification.error(YtcA, YtA)

print "Jeu de test A - Modèle LDA: erreur %s" % erreur

beta, u = classification.logistic_regression(XA, YA, verbose=False)
YtcA = classification.logistic_regression_predict(XtA, beta, u)
erreur = classification.error(YtcA, YtA)

print "Jeu de test A - Regression logisitique: erreur %s" % erreur

beta, u = classification.linear_regression(XA, YA)
YtcA = classification.linear_regression_predict(XtA, beta, u)
erreur = classification.error(YtcA, YtA)

print "Jeu de test A - Regression linéaire: erreur %s" % erreur

# Jeu de données B
print
print
print "Jeu de données B"
print "****************"
print

beta, u = classification.LDA(XB, YB)
YtcB = classification.logistic_regression_predict(XtB, beta, u)
erreur = classification.error(YtcB, YtB)

print "Jeu de test B - Modèle LDA: erreur %s" % erreur

beta, u = classification.logistic_regression(XB, YB, verbose=False)
YtcB = classification.logistic_regression_predict(XtB, beta, u)
erreur = classification.error(YtcB, YtB)

print "Jeu de test B - Regression logisitique: erreur %s" % erreur

beta, u = classification.linear_regression(XB, YB)
YtcB = classification.linear_regression_predict(XtB, beta, u)
erreur = classification.error(YtcB, YtB)

print "Jeu de test B - Regression linéaire: erreur %s" % erreur

# Jeu de données C
print
print
print "Jeu de données C"
print "****************"
print


beta, u = classification.LDA(XC, YC)
YtcC = classification.logistic_regression_predict(XtC, beta, u)
erreur = classification.error(YtcC, YtC)

print "Jeu de test C - Modèle LDA: erreur %s" % erreur

beta, u = classification.logistic_regression(XC, YC, verbose=False)
YtcC = classification.logistic_regression_predict(XtC, beta, u)
erreur = classification.error(YtcC, YtC)

print "Jeu de test C - Regression logisitique: erreur %s" % erreur

beta, u = classification.linear_regression(XC, YC)
YtcC = classification.linear_regression_predict(XtC, beta, u)
erreur = classification.error(YtcC, YtC)

print "Jeu de test C - Regression linéaire: erreur %s" % erreur


