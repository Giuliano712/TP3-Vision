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
-   Dataset trouvé sur Kaggle : 'Cards Image Dataset-Classification'

## Description de l'algorithme

Pour ce TP, nous avons décidé de travailler sur GoogleColab nottamment pour diminuer le temps d'entrainement de nos réseaux de neurones. 
Pour notre dataset, nous avons choisi les images d'un jeu de 53 cartes.
Pour l'architecture déjà entrainée, nous avons choisi VGG19. 

1. Téléchargement de notre dataset
   La première étape de notre programme est de télécharger notre dataset directement avec Kaggle.

2. Initialisation de notre modèle CNN
   On initialise notre propre CNN. Il est composé de 8 couches de convolutions

3. Création d'un fichier txt
   Nous créons un fichier txt dans lequel nous mettrons les informations utiles de nos différents entrainements.

5. Configuration de nos modèles
   On configure nos modèles de test, de train et de validation comme vu sur les tutoriels du cours.

6. Paramétrage du test
   Ici on choisi le modèle a entrainer, le nombre d'epochs, la taille des batchs et le taux d'apprentissage.

7. 
    



   
## Résultats
