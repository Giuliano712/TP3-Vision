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
- Google Colab

### Ressouces

Pour entrainer un modèle de cette envergure, nous avons décidé de passer sur **Google Colab**. En effet, nous n'avions pas tous des GPUs Nvidia et le temps d'entrainement était très long sur nos machines. Grâce aux GPU T4, nous avons pu significativement réduire nos temps d'entrainement, ce qui nous as permis d'effectuer beaucoup plus de tests et d'optimisations pour la recherche en hyperparamètres.

## Description de l'entrainement

Pour ce TP, nous avons décidé de travailler sur GoogleColab nottamment pour diminuer le temps d'entrainement de nos réseaux de neurones.
Pour notre dataset, nous avons choisi les images d'un jeu de 53 cartes.
Pour l'architecture déjà entrainée, nous avons choisi VGG19.

### Description du dataset

Notre dataset est composé de 53 classes, chacune correspondant à une carte de jeu. Les images sont de taille 224 X 224 X 3 en format jpg. Il y a en 7624 images d'entrainement, 265 images de test et 265 images de validation. Le dataset est plutôt bien équilibré.

### Description des architectures

#### Home-made CNN

Pour notre architecture CNN, nous avons créé un modèle assez basique avec **8 couches de convolution** chacune avec une activation ReLu et une normalisation. Après **une couche Flatten** pour aplatir les données, nous avons ajouté **deux couches denses** et une **fonction Softmax** pour transformer les sorties en probabilités.

Voici un visuel de notre architecture CNN :

![CNN Architecture](/Image/CNNArchitecture.png)

Cette architecture est assez petit et simple. Elle possède **13 551 365** paramètres à entraîner. Comme nous l'avons créé nous-même, nous allons faire un entrainement **from scratch**. Et tout les paramètres sont entrainables.

Nous utilisons la fonction python ci-dessous pour compter les poids du modèle :

```python
# Calculate the number of parameters
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  print_txt(f"Total parameters: {total_params}", txt_file)
  print_txt(f"Trainable parameters: {trainable_params}", txt_file)
```

#### VGG19

VGG19 est une architecture de réseau de neurones convolutifs qui a été proposée par les chercheurs du Visual Graphics Group (VGG) de l'Université d'Oxford. Il s'agit d'une version améliorée de VGG16.

Nous avons décidé de choisir VGG19 pour notre transfert learning car c'est une architecture qui a fait ses preuves et qui est assez simple à utiliser. Elle est composée de **19 couches** dont **16 couches de convolution** et **3 couches entièrement connectées**. Elle est très performante pour la classification d'images.

Voici un visuel de notre architecture VGG :

![CNN Architecture](/Image/VGGArchitecture.png)

Sachant que VGG19 est une architecture déjà entrainée sur ImageNet, nous allons faire du **transfer learning**. Nous allons **geler les premières couches** pour ne pas perdre les informations apprises sur ImageNet et **ajouter une couche dense** pour adapter le modèle à notre problème de classification de cartes. Voici le code qui illustre cette opération :

```python
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Geler tous les poids de vgg
for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, class_number)
```

En plus du VGG, nous avons ajouté une couche dense de 4096 à 53 pour notre problème de classification de cartes. Cela nous donne un total de **139 787 381**. Grâce au transfert learning, nous n'avons que **217 141** à entraîner, ce qui est bien inférieur à notre CNN.

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

- **Taille des batchs** : 16, 32, 64, 128
- **Taux d'apprentissage** : 0.0005, 0.0001, 0.00005, 0.00001
- **Weight decay** : 0.00005, 0.0001, 0.0005, 0.001

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

- **Taille des batchs** : 16, 32, 64, 128
- **Taux d'apprentissage** : 0.0005, 0.0001, 0.00005, 0.00001
- **Weight decay** : 0.00005, 0.0001, 0.0005, 0.001

##### Batch size

Nous avons laissé constant les hyperparamètres suivants :

- Taux d'apprentissage : 0.005
- Weight decay : 0.0001

| Batch Size | Accuracy  | Kappa     |
| ---------- | --------- | --------- |
| 16         | 0.230     | 0.215     |
| 32         | 0.256     | 0.242     |
| **64**     | **0.279** | **0.265** |
| 128        | 0.267     | 0.253     |

##### Learning rate

Nous avons laissé constant les hyperparamètres suivants :

- Taille de batch : 64
- Weight decay : 0.0001

| Learning Rate | Accuracy  | Kappa     |
| ------------- | --------- | --------- |
| 0.001         | 0.271     | 0.257     |
| 0.01          | 0.218     | 0.203     |
| 0.0005        | 0.252     | 0.238     |
| **0.005**     | **0.279** | **0.265** |

##### Weight decay

Nous avons laissé constant les hyperparamètres suivants :

- Taille de batch : 64
- Learning Rate : 0.005

| Weight Decay | Accuracy  |   Kappa   |
| :----------: | :-------: | :-------: |
|    5e-05     |   0.241   |   0.226   |
|    0.0005    |   0.230   |   0.215   |
|    0.001     |   0.252   |   0.238   |
|  **0.0001**  | **0.279** | **0.265** |

## Résultats

Dans cette partie, nous avons pris les meilleurs hyperparamètres pour nos deux architectures que nous avons trouvé précédemment et nous allons faire un entrainement sur 20 epochs pour comparer les performances entre les 2 modèles.

### Hyperparamètres optimaux

Ci-dessous se trouvent les hyperparamètres optimaux pour nos deux architectures :

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

#### VGG19

Sur 3 entrainements, nous avons obtenu les résultats suivants :

| Training | Accuracy | Balanced Accuracy | Kappa | Top-2 Accuracy | Top-3 Accuracy |
| -------- | -------- | ----------------- | ----- | -------------- | -------------- |
| 1        | 0.532    | 0.532             | 0.523 | 0.649          | 0.713          |
| 2        | 0.532    | 0.531             | 0.523 | 0.637          | 0.724          |
| 3        | 0.524    | 0.524             | 0.515 | 0.664          | 0.717          |
| Moyenne  | 0,529    | 0,529             | 0,520 | 0,65           | 0,718          |

#### CNN

Sur 3 entrainements, nous avons obtenu les résultats suivants :

| Training | Accuracy | Balanced Accuracy | Kappa | Top-2 Accuracy | Top-3 Accuracy |
| -------- | -------- | ----------------- | ----- | -------------- | -------------- |
| 1        | 0.415    | 0.415             | 0.403 | 0.445          | 0.471          |
| 2        | 0.384    | 0.384             | 0.373 | 0.437          | 0.449          |
| 3        | 0.392    | 0.392             | 0.380 | 0.426          | 0.456          |
| Moyenne  | 0,397    | 0,397             | 0,385 | 0,436          | 0,459          |

### Visualisation des entrainements

#### Loss

Les courbes de loss montrent que les deux entraînements ont convergé efficacement, avec une diminution régulière des valeurs de perte au fil des itérations.

Cependant, on observe que le modèle VGG19 atteint une loss finale beaucoup plus basse (0,2) par rapport au CNN basique (3,56). Cette différence s'explique par le fait que VGG19 a été initialisé avec des poids pré-entraînés sur le dataset ImageNet, lui conférant une meilleure capacité de généralisation dès le départ. En revanche, le CNN a été entraîné à partir de zéro, sans bénéficier d'une pré-entraînement.

Cette observation suggère que, bien que le VGG19 soit plus performant dans ce contexte, il pourrait être intéressant de prolonger l'entraînement du CNN pour évaluer s'il continue à réduire la loss et à se rapprocher des performances de VGG19.

![Loss CNN](/Image/LossCNN.png)
![Loss VGG](/Image/LossVGG.png)

#### Accuracy

Les courbes d'accuracy mettent en évidence que le VGG19 surpasse largement le CNN en termes de performance. Le VGG19 atteint une accuracy finale de 0,93, tandis que le CNN se limite à 0,44.

On observe également que la courbe d'accuracy du VGG19 se stabilise, ce qui indique que son entraînement est arrivé à maturation. En revanche, la courbe du CNN n'est pas encore stabilisée, ce qui suggère qu'un entraînement supplémentaire pourrait potentiellement améliorer ses performances.

Il est toutefois important de noter un phénomène de surapprentissage (overfitting) dans le transfert d'apprentissage du VGG19. L'accuracy sur l'ensemble d'entraînement est de 0,93, alors qu'elle chute à 0,53 sur l'ensemble de test. À l'opposé, les courbes d'entraînement et de validation du CNN sont plus proches, ce qui indique une meilleure cohérence et moins de surapprentissage, bien que ses performances globales soient inférieures à celles du VGG19.

![Accuracy CNN](/Image/AccuracyCNN.png)
![Accuracy VGG](/Image/AccuracyVGG.png)

#### Kappa

Les courbes de Kappa montrent des tendances similaires à celles des courbes d'accuracy. Le VGG19 obtient un Kappa final de 0,92, tandis que le CNN se limite à 0,40.

![Kappa CNN](/Image/KappaCNN.png)
![Kappa VGG](/Image/KappaVGG.png)

#### Top-3 accuracy

Les courbes de Top-3 accuracy montrent que le VGG19 surpasse également le CNN en termes de performance. Le VGG19 atteint une Top-3 accuracy finale de 1, tandis que le CNN se limite à 0,46.

![Top-3-accuracy CNN](/Image/Top-3-accuracyCNN.png)
![Top-3-accuracy VGG](/Image/Top-3-accuracyVGG.png)

## Conclusion

Ce TP nous a permis de comprendre l'utilité du transfer learning d'une architecture déjà entrainée comparée à un modèle CNN qui s'entraine de zéro. La première méthode obtient des meilleurs accuracy avec une meilleure rapidité en sachant que le model VGG19 s'entraine sur 217 141 paramètres alors que notre CNN est sur 13 551 365 paramètres.
Nous avons également compris l'influence des hyperparamètres sur l'accuracy de nos modèles.
