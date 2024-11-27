# TP3 Vision artificielle et traitement des images

## Étudiants

- **Goudal** Victor GOUV07120100
- **Erard** Julien ERAJ09050200
- **Flaceliere** Matthieu FLAM30090200

## Introduction

Ce rapport présente notre travail sur le TP3 du cours 8INF804 - Vision artificielle et traitement des images. L'objectif est de comparer deux approches dans les réseaux de neurones. Le premier est de faire du transfer learning avec une architecture connue (ici VGG19) sur un ensemble de données choisi. La deuxième est de construire notre propre architecture CNN et de classifier l'ensemble de données avec celle-ci.

## Utilisation

### Prérequis

- Python 3
- Libs: pandas, os, pytorch, numpy, tqdm, sklearn.metrics, plotly.express, sys
- Dataset trouvé sur Kaggle : [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

## Description de l'entrainement

Pour ce TP, nous avons décidé de travailler sur GoogleColab nottamment pour diminuer le temps d'entrainement de nos réseaux de neurones.
Pour notre dataset, nous avons choisi les images d'un jeu de 53 cartes.
Pour l'architecture déjà entrainée, nous avons choisi VGG19.

### Description du dataset

Notre dataset est composé de 53 classes, chacune correspondant à une carte de jeu. Les images sont de taille 224 X 224 X 3 en format jpg. Il y a en 7624 images d'entrainement, 265 images de test et 265 images de validation. Le dataset est plutôt bien équilibré.

### Description des architectures

#### VGG19

#### Home-made CNN

Pour notre architecture CNN, nous avons créé un modèle assez basique avec 8 couches de convolution chacune avec une activation ReLu et une normalisation. Après une couche Flatten pour aplatir les données, nous avons ajouté deux couches denses et une fonction Softmax pour transformer les sorties en probabilités.

Voici un visuel de notre architecture CNN :

![CNN Architecture](/Image/CNNArchitecture.png)

### Description de l'algorithme

Dans cette partie, nous allons les différentes fonctions qui compose l'entrainement de notre modèle.

#### Transformations des images

Le pré-traitements des images est une étape importante très importante de l'entrainement. Dans notre cas, nous avons décidé de ne pas faire de la data augmentation et nous n'avons pas normaliser car nous n'obtenions pas de meilleurs résultats.

Voici les transformations que nous avons appliquées sur les images :

- **Resize** : 256x256 pour VGG19 et 224x224 pour notre CNN.
- **ToTensor** : pour transformer les images en tenseurs.

#### Optimizer et loss function

Nous avons choisi d'utiliser l'optimiseur **AdamW** pour entraînement le modèle en raison de ses avantages significatifs par rapport à Adam et SGD. Contrairement à Adam, AdamW applique une **désintégration des poids** (weight decay) correcte en la séparant explicitement du terme de mise à jour des gradients, ce qui **améliore la généralisation du modèle**. Cette distinction est **essentielle pour éviter l'overfitting**, surtout sur des modèles complexes comme les CNN.

En comparaison avec SGD, **AdamW (comme Adam) converge plus rapidement** grâce à son mécanisme adaptatif de mise à jour des pas d'apprentissage, ce qui est particulièrement utile lorsque le problème implique des données complexes ou des gradients bruités. De plus, **AdamW (comme Adam) conserve une meilleure stabilité en présence de plateaux dans la fonction de perte**, ce qui est courant lors de l'entraînement de CNN.

## Méthodologie de recherche et d'expérimentation

Après avoir créé notre architecture CNN et choisi VGG19 pour le transfert learning. Il nous faut une stratégie pour optimiser nos architectures pour mieux les comparer sur notre dataset d'entrainement.

### Recherche des hyperparamètres

Pour notre architecture CNN et VGG19, les hyperparamètres sont les suivants : `batch_size`, `learning_rate`, `weight_decay`, `epochs`.

Avant de modifier les epochs nous avons voulu trouver les meilleurs valeurs pour les autres hyperparamètres. Pour trouver les meilleurs valeurs, nous avons fait varier les valeurs de ces hyperparamètres un par un et avons regardé les résultats obtenus (notamment l'accuracy car notre dataset est équilibré).

#### CNN

Pour l'entrainement de notre modèle CNN, nous avons fait varier les hyperparamètres suivants :

- Taille des batchs : 16, 32, 64, 128
- Taux d'apprentissage : 0.0005, 0.0001, 0.00005, 0.00001
- Weight decay : 0.00005, 0.0001, 0.0005, 0.001

Pour ne pas trop consommer de ressources et gagner du temps, nous avons fait nos recherches sur 3 epochs.

##### Batch size

Nous avons laissé constant les hyperparamètres suivants :

- Taux d'apprentissage : 5e-5
- Weight decay : 1e-4

| Batch Size | Accuracy  | Kappa     |
| ---------- | --------- | --------- |
| 16         | 0.245     | 0.230     |
| 32         | 0.290     | 0.276     |
| **64**     | **0.335** | **0.323** |
| 128        | 0.275     | 0.261     |

##### Learning rate

Nous avons laissé constant les hyperparamètres suivants :

- Taille de batch : 64
- Weight decay : 1e-4

| Learning Rate | Accuracy   | Kappa      |
| ------------- | ---------- | ---------- |
| 0.00001       | 0.2755     | 0.2615     |
| **0.00005**   | **0.3358** | **0.3231** |
| 0.0001        | 0.2641     | 0.2500     |
| 0.0005        | 0.0189     | 0.0000     |

##### Weight decay

Nous avons laissé constant les hyperparamètres suivants :

- Taille de batch : 64
- Learning Rate : 5e-5

| Weight Decay |  Accuracy  |   Kappa    |
| :----------: | :--------: | :--------: |
|   0.00005    |   0.3132   |   0.3000   |
|  **0.0001**  | **0.3358** | **0.3231** |
|    0.0005    |   0.2792   |   0.2654   |
|    0.001     |   0.3208   |   0.3077   |

#### VGG19

Pour l'entrainement de VGG nous avons fait varier les mêmes hyperparamètres que pour notre CNN.

- Taille des batchs : 16, 32, 64, 128
- Taux d'apprentissage : 0.0005, 0.0001, 0.00005, 0.00001
- Weight decay : 0.0001, 0.0005, 0.001

##### Batch size

Nous avons laissé constant les hyperparamètres suivants :

- Taux d'apprentissage : 0.005
- Weight decay : 0.0001

| Batch Size | Accuracy | Kappa |
| ---------- | -------- | ----- |
| 16         | 0.230    | 0.215 |
| 32         | 0.256    | 0.242 |
| 64         | 0.279    | 0.265 |
| 128        | 0.267    | 0.253 |

##### Learning rate

Nous avons laissé constant les hyperparamètres suivants :

- Taille de batch : 64
- Weight decay : 0.0001

| Learning Rate | Accuracy | Kappa |
| ------------- | -------- | ----- |
|               |          |       |
|               |          |       |
|               |          |       |
|               |          |       |

##### Weight decay

Nous avons laissé constant les hyperparamètres suivants :

- Taille de batch : 64
- Learning Rate : 5e-5

| Weight Decay | Accuracy | Kappa |
| :----------: | :------: | :---: |
|              |          |       |
|              |          |       |
|              |          |       |
|              |          |       |

## Résultats

### Hyperparamètres optimaux

#### VGG19

- `batch_size` : 64
- `learning_rate` : 0.005
- `weight_decay` : 0.0001
- `epochs` : 20

#### CNN

- `batch_size` : 64
- `learning_rate` : 0.00005
- `weight_decay` : 0.0001
- `epochs` : 20

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
