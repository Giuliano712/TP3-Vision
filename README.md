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

CNNClassifier(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
    (4): ReLU()
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (7): ReLU()
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
    (10): ReLU()
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (13): ReLU()
    (14): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (15): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
    (16): ReLU()
    (17): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (18): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
    (19): ReLU()
    (20): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
    (22): ReLU()
    (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (24): Flatten(start_dim=1, end_dim=-1)
    (25): Linear(in_features=73728, out_features=120, bias=True)
    (26): ReLU()
    (27): Linear(in_features=120, out_features=84, bias=True)
    (28): ReLU()
    (29): Linear(in_features=84, out_features=53, bias=True)
    (30): Softmax(dim=1)
  )
)
Total parameters: 13551365
Trainable parameters: 13551365

### Description de l'algorithme

#### Transformations des images

#### Optimizer et loss function

## Méthodologie de recherche et d'expérimentation

Après avoir créé notre architecture CNN et choisi VGG19 pour le transfert learning. Il nous faut une stratégie pour optimiser nos architectures pour mieux les comparer sur notre dataset d'entrainement.

### Recherche des hyperparamètres

Pour notre architecture CNN et VGG19, les hyperparamètres sont les suivants : `batch_size`, `learning_rate`, `weight_decay`, `epochs`.

Avant de modifier les epochs nous avons voulu trouver les meilleurs valeurs pour les autres hyperparamètres. Pour trouver les meilleurs valeurs, nous avons fait varier les valeurs de ces hyperparamètres un par un et avons regardé les résultats obtenus (notamment l'accuracy car notre dataset est équilibré).

#### CNN

Pour l'entrainement de notre modèle CNN, nous avons fait varier les hyperparamètres suivants :
-  Taille des batchs : 16, 32, 64, 128
-  Taux d'apprentissage : 0.0005, 0.0001, 0.00005, 0.00001
-  Weight decay : 0.0001, 0.0005, 0.001

Pour ne pas trop consommer de ressources et gagner du temps, nous avons fait nos recherches sur 3 epochs.

##### Batch size

Nous avons laissé constant les hyperparamètres suivants :
-  Taux d'apprentissage : 5e-5
-  Weight decay : 1e-4

| Batch Size | Balanced Accuracy | Kappa          |
|------------|--------------------|----------------|
| 16         | 0.245 | 0.230 |
| 32         | 0.290  | 0.276 |
| **64**         | **0.335**  | **0.323** |
| 128        | 0.275  | 0.261  |


##### Learning rate

Nous avons laissé constant les hyperparamètres suivants :
-  Taille de batch : 64
-  Weight decay : 1e-4

| Learning Rate | Balanced Accuracy | Kappa     |
|---------------|--------------------|-----------|
| 0.00001       | 0.2755            | 0.2615    |
| **0.00005**       | **0.3358**            | **0.3231**    |
| 0.0001        | 0.2641            | 0.2500    |
| 0.0005        | 0.0189            | 0.0000    |

##### Weight decay

Nous avons laissé constant les hyperparamètres suivants :
-  Taille de batch : 64
-  Learning Rate : 5e-5

| Weight Decay | Balanced Accuracy | Kappa     |
|:------------:|:-----------------:|:---------:|
| 0.00005      | 0.3132            | 0.3000    |
| 0.0001       | 0.3358            | 0.3231    |
| 0.0005       | 0.2792            | 0.2654    |
| 0.001        | 0.3208            | 0.3077    |





#### VGG19

Les valeurs que nous avons testées pour VGG19 sont les suivantes :

- `batch_size` : 32,64,128
- `learning_rate` : 0.001, 0.0025, 0.005, 0.01
- `weight_decay` : 0.0001, 0.0005, 0.1
- `epochs` : 3, 5, 10, 20

#### Home-made CNN

Les valeurs que nous avons testées pour notre CNN sont les suivantes :

- `batch_size` : 32,64,128
- `learning_rate` : 0.00001, 0.00005, 0.0005, 0.0001, 0.005
- `weight_decay` : 0.00001, 0.00005, 0.0001
- `epochs` : 3, 5, 10, 20

## Résultats

### Hyperparamètres optimaux

#### VGG19

- `batch_size` : 64
- `learning_rate` : 0.005
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
