# TP3 Vision artificielle et traitement des images

## Étudiants

-   **Goudal** Victor GOUV07120100
-   **Erard** Julien ERAJ09050200
-   **Flaceliere** Matthieu FLAM30090200

## Introduction

Ce rapport présente notre travail sur le TP3 du cours 8INF804 - Vision artificielle et traitement des images. L'objectif est de comparer deux approches dans les réseaux de neurones. Le premier est de faire du transfer learning avec une architecture connue (ici VGG19) sur un ensemble de données choisi. La deuxième est de construire notre propre architecture CNN et de classifier l'ensemble de données avec celle-ci.

## Utilisation

### Prérequis

-   Python 3
-   Libs: pandas, os, pytorch, numpy, tqdm, sklearn.metrics, plotly.express, sys
-   Dataset trouvé sur Kaggle : [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

## Description de l'entrainement

Pour ce TP, nous avons décidé de travailler sur GoogleColab nottamment pour diminuer le temps d'entrainement de nos réseaux de neurones. 
Pour notre dataset, nous avons choisi les images d'un jeu de 53 cartes.
Pour l'architecture déjà entrainée, nous avons choisi VGG19.

### Description du dataset

### Description des architectures

#### VGG19

#### Home-made CNN

### Description de l'algorithme

#### Transformations des images

#### Optimizer et loss function

## Méthodologie de recherche et d'expérimentation

### Recherche des hyperparamètres

### Entrainements sur beaucoup d'epochs

## Résultats

### Hyperparamètres optimaux

### Evaluation des performances

### Visualisation des entrainements

## Conclusion



1. Téléchargement de notre dataset

La première étape de notre programme est de télécharger notre dataset directement avec Kaggle.

2. Initialisation de notre modèle CNN

On initialise notre propre CNN. Il est composé de 8 couches de convolutions

3. Functions



4. Configuration de nos modèles

On configure nos modèles de test, de train et de validation comme vu sur les tutoriels du cours.

5. Paramétrage du test

Ici on choisi le modèle a entrainer, le nombre d'epochs, la taille des batchs et le taux d'apprentissage. Pour ces entrainement, nous utilisons l'optimiseur Adam.

6. Création des logs

Nous crééons un fichier txt dans lequel nous mettons les paramètres du test et nous mettrons les données du test comme l'accuracy, le coeff de kappa et les courbes d'apprentissage.

7. Charger le dataset

Nous transformons le dataset en fonction du modèle utilisé (VGG19 ou notre CNN).
Puis nous chargeons le dataset dans les modèles d'entrainement, de test et de validation.

8. Lancement de l'entrainement

Nous choisissons 
    

   
## Résultats
