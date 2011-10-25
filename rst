Jeu de données A
****************

Jeu de test A - Modèle LDA: erreur 0.216666666667
* Jeu de test A - Regression logisitique: erreur 0.16
* Jeu de test A - Regression linéaire: erreur 0.16


Jeu de données B
****************

Jeu de test B - Modèle LDA: erreur 0.126666666667
Jeu de test B - Regression logisitique: erreur 0.1
* Jeu de test B - Regression linéaire: erreur 0.0833333333333


Jeu de données C
****************

Jeu de test C - Modèle LDA: erreur 0.155
* Jeu de test C - Regression logisitique: erreur 0.12
Jeu de test C - Regression linéaire: erreur 0.16


La régression linéaire est très peu robuste (Jeu de données C - label bimodale
pour Y = 0;
jeu de données B - densité à décroissance à rapide et à matrice de covariance
non scalaire pour Y = 1).
=> ne marche que pour des données unimodales

On observe une grande ressemblance dans les résultats de la regression
linéaire et logisitique (jeu de tests A).

Jeu de tests A

Jeu de tests B
La régression linéaire fonctionne très bien sur ce jeu de données, car les
points tels que Y = 1 sont visuellement proches d'une droite, qui est elel
même orthogonal à la moyenne des vecteurs pour Y = 0. Ainsi il semble réaliste
de trouver un vecteur theta tel que pour le point X tel que Y = 1, theta X =
1, et dans l'autre cas theta.X = 0

On remarque que les deux autres méthodes sont aussi relativement bonnes. Ce
n'est pas surprenant, les données étant visuellement séparées.

Jeu de tests C
Les points sont visuellement proche d'une même droite. Si un vecteur theta est
tel que theta.X = 1 pour Y = 1, alors pour une grande partie des points tels
que Y = 0, on aura aussi theta X = 1.
Pour la régression linéaire, la distance au barycentre des points joue
énormement (mauvais traitement des données bimodales). Même les points
exceptionnellement éloignés du barycentre ont une importance sur le calcul de
theta, alors qu'ils représentent plus du bruit.

De même que pour le jeu de données B, IRLS et LDA donnent des résultats tout à
fait satisfaisants.

Jeu de données A
Le barycentre des données est équidistant des nuages de points Y = 0 et Y = 1,
ce qui explique la bonne performance de la régression linéaire.

Entre LDA et IRLS, la pente des droites est très proche, mais la constante
varie. IRLS est une méthode d'approximation itérative, qui suit le même
principe que LMS (least m square), qui converge très bien pour la régression
linéaire. Comem cette dernière est très efficace ici, il n'est donc pas
suprenant qu'IRLS marche si bien sur ces données.

On peut supposer que si il y a un biais dans les données, il sera plus
facilement corrigé par un algorithme iteratif (IRLS) que par un calcul direct
(LDA). Cela peut expliquer le plus faible taux d'erreur pour IRLS que pour
LDA.


