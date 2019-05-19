

Code à utiliser pour faire le clustering des descripteurs des images


Dépendances:

PQk-means : pip install pqkmeans
scikit-learn : pip install scikit-learn
numpy : pip install numpy


Utilisation:

Pour la création de l'index, utiliser la fonction dataset_handling du fichier create_clusters.py.
Cette fonction retourne un dictionnaire dont les clés sont les différents noms d'images et les valeurs une liste des mots visuels.
Ceux-ci sont ensuite utilisés pour la création de l'index comme pour les textes.

Pour la gestion de la requête, utiliser la fonction request_handling dans le fichier request.py.
A partir d'un nom d'image, cette fonction retourne la liste des mots visuels pour cette image.


