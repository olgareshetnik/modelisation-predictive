
Projet: "Détection des faux billets avec Python"
Contexte du projet

<> L’Organisation nationale de lutte contre le faux-monnayage ( ONCFM), est une organisation publique ayant pour objectif de mettre en place des méthodes d’identification des contrefaçons des billets en euros. Dans le cadre de cette lutte, l’Organisation souhaite mettre en place un algorithme qui soit capable de différencier automatiquement les vrais des faux billets <>
Objectifs

<> Lorsqu’un billet arrive, nous avons une machine qui consigne l’ensemble de ses caractéristiques géométriques. Tout au long des années de lutte, nous avons observé des différences de dimensions entre les vrais et les faux billets. Ces différences sont difficilement notables à l’œil nu, mais une machine devrait sans problème arriver à les différencier. Ainsi, il faudrait construire un algorithme qui, à partir des caractéristiques géométriques d’un billet, serait capable de définir si ce dernier est un vrai ou un faux billet <>.
Fonctionnement général

<>Nous avons à notre disposition six données géométriques pour chaque billet. L’algorithme devra donc être capable de prendre en entrée un fichier contenant les dimensions de plusieurs billets, et de déterminer le type de chacun d’entre eux, à partir des seules dimensions<>

# On importe les librairies dont on aura besoin pour la  modélisation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# On charge le fichier 'billets'
billets = pd.read_csv('billets.csv')
billets.head()

	is_genuine 	diagonal 	height_left 	height_right 	margin_low 	margin_up 	length
0 	True 	171.81 	104.86 	104.95 	4.52 	2.89 	112.83
1 	True 	171.46 	103.36 	103.66 	3.77 	2.99 	113.09
2 	True 	172.69 	104.48 	103.50 	4.40 	2.94 	113.16
3 	True 	171.36 	103.91 	103.94 	3.62 	3.01 	113.51
4 	True 	171.73 	104.28 	103.46 	4.04 	3.48 	112.54


1. Préparation des données pour l'analyse
Une analyse descriptive des données

billets.shape

(1500, 7)

n_vrais_faux = billets.is_genuine.value_counts()
print(n_vrais_faux)

True     1000
False     500
Name: is_genuine, dtype: int64

billets.nunique()

is_genuine        2
diagonal        159
height_left     155
height_right    170
margin_low      285
margin_up       123
length          336
dtype: int64

billets.duplicated().sum()

0

# détecter valeur manquent
billets.isna().sum()

is_genuine       0
diagonal         0
height_left      0
height_right     0
margin_low      37
margin_up        0
length           0
dtype: int64

Je vais remplacer les valeurs manquantes par des valeurs prédites en utilisent Regression Lineare

# analyse descriptive des données : 
billets.describe()

	diagonal 	height_left 	height_right 	margin_low 	margin_up 	length
count 	1500.000000 	1500.000000 	1500.000000 	1463.000000 	1500.000000 	1500.00000
mean 	171.958440 	104.029533 	103.920307 	4.485967 	3.151473 	112.67850
std 	0.305195 	0.299462 	0.325627 	0.663813 	0.231813 	0.87273
min 	171.040000 	103.140000 	102.820000 	2.980000 	2.270000 	109.49000
25% 	171.750000 	103.820000 	103.710000 	4.015000 	2.990000 	112.03000
50% 	171.960000 	104.040000 	103.920000 	4.310000 	3.140000 	112.96000
75% 	172.170000 	104.230000 	104.150000 	4.870000 	3.310000 	113.34000
max 	173.010000 	104.880000 	104.950000 	6.900000 	3.910000 	114.44000
Séparer les ligne avec des valeurs NULL et considérez-les comme 'test data'

test_data=billets[billets["margin_low"].isnull()]
test_data

	is_genuine 	diagonal 	height_left 	height_right 	margin_low 	margin_up 	length
72 	True 	171.94 	103.89 	103.45 	NaN 	3.25 	112.79
99 	True 	171.93 	104.07 	104.18 	NaN 	3.14 	113.08
151 	True 	172.07 	103.80 	104.38 	NaN 	3.02 	112.93
197 	True 	171.45 	103.66 	103.80 	NaN 	3.62 	113.27
241 	True 	171.83 	104.14 	104.06 	NaN 	3.02 	112.36
251 	True 	171.80 	103.26 	102.82 	NaN 	2.95 	113.22
284 	True 	171.92 	103.83 	103.76 	NaN 	3.23 	113.29
334 	True 	171.85 	103.70 	103.96 	NaN 	3.00 	113.36
410 	True 	172.56 	103.72 	103.51 	NaN 	3.12 	112.95
413 	True 	172.30 	103.66 	103.50 	NaN 	3.16 	112.95
445 	True 	172.34 	104.42 	103.22 	NaN 	3.01 	112.97
481 	True 	171.81 	103.53 	103.96 	NaN 	2.71 	113.99
505 	True 	172.01 	103.97 	104.05 	NaN 	2.98 	113.65
611 	True 	171.80 	103.68 	103.49 	NaN 	3.30 	112.84
654 	True 	171.97 	103.69 	103.54 	NaN 	2.70 	112.79
675 	True 	171.60 	103.85 	103.91 	NaN 	2.56 	113.27
710 	True 	172.03 	103.97 	103.86 	NaN 	3.07 	112.65
739 	True 	172.07 	103.74 	103.76 	NaN 	3.09 	112.41
742 	True 	172.14 	104.06 	103.96 	NaN 	3.24 	113.07
780 	True 	172.41 	103.95 	103.79 	NaN 	3.13 	113.41
798 	True 	171.96 	103.84 	103.62 	NaN 	3.01 	114.44
844 	True 	171.62 	104.14 	104.49 	NaN 	2.99 	113.35
845 	True 	172.02 	104.21 	104.05 	NaN 	2.90 	113.62
871 	True 	171.37 	104.07 	103.75 	NaN 	3.07 	113.27
895 	True 	171.81 	103.68 	103.80 	NaN 	2.98 	113.82
919 	True 	171.92 	103.68 	103.45 	NaN 	2.58 	113.68
945 	True 	172.09 	103.74 	103.52 	NaN 	3.02 	112.78
946 	True 	171.63 	103.87 	104.66 	NaN 	3.27 	112.68
981 	True 	172.02 	104.23 	103.72 	NaN 	2.99 	113.37
1076 	False 	171.57 	104.27 	104.44 	NaN 	3.21 	111.87
1121 	False 	171.40 	104.38 	104.19 	NaN 	3.17 	112.39
1176 	False 	171.59 	104.05 	103.94 	NaN 	3.02 	111.29
1303 	False 	172.17 	104.49 	103.76 	NaN 	2.93 	111.21
1315 	False 	172.08 	104.15 	104.17 	NaN 	3.40 	112.29
1347 	False 	171.72 	104.46 	104.12 	NaN 	3.61 	110.31
1435 	False 	172.66 	104.33 	104.41 	NaN 	3.56 	111.47
1438 	False 	171.90 	104.28 	104.29 	NaN 	3.24 	111.49

test_data.shape

(37, 7)

Supprimer les ligne avec des valeurs NULL de DataFrame et considérez-les comme 'train data'

billets.dropna(inplace=True)
billets

	is_genuine 	diagonal 	height_left 	height_right 	margin_low 	margin_up 	length
0 	True 	171.81 	104.86 	104.95 	4.52 	2.89 	112.83
1 	True 	171.46 	103.36 	103.66 	3.77 	2.99 	113.09
2 	True 	172.69 	104.48 	103.50 	4.40 	2.94 	113.16
3 	True 	171.36 	103.91 	103.94 	3.62 	3.01 	113.51
4 	True 	171.73 	104.28 	103.46 	4.04 	3.48 	112.54
... 	... 	... 	... 	... 	... 	... 	...
1495 	False 	171.75 	104.38 	104.17 	4.42 	3.09 	111.28
1496 	False 	172.19 	104.63 	104.44 	5.27 	3.37 	110.97
1497 	False 	171.80 	104.01 	104.12 	5.51 	3.36 	111.95
1498 	False 	172.06 	104.28 	104.06 	5.17 	3.46 	112.25
1499 	False 	171.47 	104.15 	103.82 	4.63 	3.37 	112.07

1463 rows × 7 columns

# la taille du tableau
billets.shape

(1463, 7)

# vérifier les valeurs manquantes
billets.isnull().sum()

is_genuine      0
diagonal        0
height_left     0
height_right    0
margin_low      0
margin_up       0
length          0
dtype: int64

Créer x_train et y_train à partir de nouveau DataFrame

# nous séparons la colonne 'margin_low' pour 'y_train'
y_train=billets["margin_low"]

y_train

0       4.52
1       3.77
2       4.40
3       3.62
4       4.04
        ... 
1495    4.42
1496    5.27
1497    5.51
1498    5.17
1499    4.63
Name: margin_low, Length: 1463, dtype: float64

# la taille de data set
y_train.shape

(1463,)

# et crée 'x_train' - dataset sans "margin_low" 
x_train=billets.drop("margin_low",axis=1)

# la taille de data set
x_train.shape

(1463, 6)

Construire le modèle

-> préparer le modèle (Lineare Regression) pour prédire les valeurs manquantes dans la colonne "margin_low"

# import Linear Regression depuit sklearn 
from sklearn.linear_model import LinearRegression

# créer un modèle Regression Linear 
linr = LinearRegression()

# entrainer model en data set 'train'
linr.fit(x_train,y_train)

LinearRegression()

Créer x_test a partire de test_data

# 'x_test' signifie dataset sauf "margin_low"  avec les valeures NULL
X_test=test_data.drop("margin_low",axis=1)
X_test

	is_genuine 	diagonal 	height_left 	height_right 	margin_up 	length
72 	True 	171.94 	103.89 	103.45 	3.25 	112.79
99 	True 	171.93 	104.07 	104.18 	3.14 	113.08
151 	True 	172.07 	103.80 	104.38 	3.02 	112.93
197 	True 	171.45 	103.66 	103.80 	3.62 	113.27
241 	True 	171.83 	104.14 	104.06 	3.02 	112.36
251 	True 	171.80 	103.26 	102.82 	2.95 	113.22
284 	True 	171.92 	103.83 	103.76 	3.23 	113.29
334 	True 	171.85 	103.70 	103.96 	3.00 	113.36
410 	True 	172.56 	103.72 	103.51 	3.12 	112.95
413 	True 	172.30 	103.66 	103.50 	3.16 	112.95
445 	True 	172.34 	104.42 	103.22 	3.01 	112.97
481 	True 	171.81 	103.53 	103.96 	2.71 	113.99
505 	True 	172.01 	103.97 	104.05 	2.98 	113.65
611 	True 	171.80 	103.68 	103.49 	3.30 	112.84
654 	True 	171.97 	103.69 	103.54 	2.70 	112.79
675 	True 	171.60 	103.85 	103.91 	2.56 	113.27
710 	True 	172.03 	103.97 	103.86 	3.07 	112.65
739 	True 	172.07 	103.74 	103.76 	3.09 	112.41
742 	True 	172.14 	104.06 	103.96 	3.24 	113.07
780 	True 	172.41 	103.95 	103.79 	3.13 	113.41
798 	True 	171.96 	103.84 	103.62 	3.01 	114.44
844 	True 	171.62 	104.14 	104.49 	2.99 	113.35
845 	True 	172.02 	104.21 	104.05 	2.90 	113.62
871 	True 	171.37 	104.07 	103.75 	3.07 	113.27
895 	True 	171.81 	103.68 	103.80 	2.98 	113.82
919 	True 	171.92 	103.68 	103.45 	2.58 	113.68
945 	True 	172.09 	103.74 	103.52 	3.02 	112.78
946 	True 	171.63 	103.87 	104.66 	3.27 	112.68
981 	True 	172.02 	104.23 	103.72 	2.99 	113.37
1076 	False 	171.57 	104.27 	104.44 	3.21 	111.87
1121 	False 	171.40 	104.38 	104.19 	3.17 	112.39
1176 	False 	171.59 	104.05 	103.94 	3.02 	111.29
1303 	False 	172.17 	104.49 	103.76 	2.93 	111.21
1315 	False 	172.08 	104.15 	104.17 	3.40 	112.29
1347 	False 	171.72 	104.46 	104.12 	3.61 	110.31
1435 	False 	172.66 	104.33 	104.41 	3.56 	111.47
1438 	False 	171.90 	104.28 	104.29 	3.24 	111.49

X_test.shape

(37, 6)

Appliquer le modèle entraîné pour prédire les valeurs manquantes

# appliquer le modèle trained sur x_test
y_pred=linr.predict(X_test)
y_pred

array([4.06495361, 4.11199026, 4.13400328, 3.99357074, 4.1403993 ,
       4.09428392, 4.07412432, 4.12538999, 4.0807278 , 4.07363322,
       4.11897255, 4.18037978, 4.13648423, 4.05106842, 4.17837685,
       4.22555104, 4.11586845, 4.10284101, 4.08184346, 4.09276238,
       4.11250192, 4.15717623, 4.16028787, 4.12193808, 4.12353555,
       4.19842271, 4.10962313, 4.09696025, 4.13384101, 5.25968515,
       5.264817  , 5.28251853, 5.30206887, 5.20035843, 5.1754678 ,
       5.17345045, 5.24675055])

y_pred.shape

(37,)

Remplacer les valeurs manquantes par des valeurs prédites

test_data.loc[test_data.margin_low.isnull(), "margin_low"]=y_pred

test_data

	is_genuine 	diagonal 	height_left 	height_right 	margin_low 	margin_up 	length
72 	True 	171.94 	103.89 	103.45 	4.064954 	3.25 	112.79
99 	True 	171.93 	104.07 	104.18 	4.111990 	3.14 	113.08
151 	True 	172.07 	103.80 	104.38 	4.134003 	3.02 	112.93
197 	True 	171.45 	103.66 	103.80 	3.993571 	3.62 	113.27
241 	True 	171.83 	104.14 	104.06 	4.140399 	3.02 	112.36
251 	True 	171.80 	103.26 	102.82 	4.094284 	2.95 	113.22
284 	True 	171.92 	103.83 	103.76 	4.074124 	3.23 	113.29
334 	True 	171.85 	103.70 	103.96 	4.125390 	3.00 	113.36
410 	True 	172.56 	103.72 	103.51 	4.080728 	3.12 	112.95
413 	True 	172.30 	103.66 	103.50 	4.073633 	3.16 	112.95
445 	True 	172.34 	104.42 	103.22 	4.118973 	3.01 	112.97
481 	True 	171.81 	103.53 	103.96 	4.180380 	2.71 	113.99
505 	True 	172.01 	103.97 	104.05 	4.136484 	2.98 	113.65
611 	True 	171.80 	103.68 	103.49 	4.051068 	3.30 	112.84
654 	True 	171.97 	103.69 	103.54 	4.178377 	2.70 	112.79
675 	True 	171.60 	103.85 	103.91 	4.225551 	2.56 	113.27
710 	True 	172.03 	103.97 	103.86 	4.115868 	3.07 	112.65
739 	True 	172.07 	103.74 	103.76 	4.102841 	3.09 	112.41
742 	True 	172.14 	104.06 	103.96 	4.081843 	3.24 	113.07
780 	True 	172.41 	103.95 	103.79 	4.092762 	3.13 	113.41
798 	True 	171.96 	103.84 	103.62 	4.112502 	3.01 	114.44
844 	True 	171.62 	104.14 	104.49 	4.157176 	2.99 	113.35
845 	True 	172.02 	104.21 	104.05 	4.160288 	2.90 	113.62
871 	True 	171.37 	104.07 	103.75 	4.121938 	3.07 	113.27
895 	True 	171.81 	103.68 	103.80 	4.123536 	2.98 	113.82
919 	True 	171.92 	103.68 	103.45 	4.198423 	2.58 	113.68
945 	True 	172.09 	103.74 	103.52 	4.109623 	3.02 	112.78
946 	True 	171.63 	103.87 	104.66 	4.096960 	3.27 	112.68
981 	True 	172.02 	104.23 	103.72 	4.133841 	2.99 	113.37
1076 	False 	171.57 	104.27 	104.44 	5.259685 	3.21 	111.87
1121 	False 	171.40 	104.38 	104.19 	5.264817 	3.17 	112.39
1176 	False 	171.59 	104.05 	103.94 	5.282519 	3.02 	111.29
1303 	False 	172.17 	104.49 	103.76 	5.302069 	2.93 	111.21
1315 	False 	172.08 	104.15 	104.17 	5.200358 	3.40 	112.29
1347 	False 	171.72 	104.46 	104.12 	5.175468 	3.61 	110.31
1435 	False 	172.66 	104.33 	104.41 	5.173450 	3.56 	111.47
1438 	False 	171.90 	104.28 	104.29 	5.246751 	3.24 	111.49
Évaluation du modèle

from sklearn.metrics import r2_score

Y_test=test_data["margin_low"]

# score R carré du modèle
r2_ = r2_score(Y_test, y_pred)
print("La performance du Modèle pour le set de Test")
print("--------------------------------------------")
print('le score R2 score est {}'.format(r2_))

La performance du Modèle pour le set de Test
--------------------------------------------
le score R2 score est 1.0

test_data.shape

(37, 7)

billets.isna().sum()

is_genuine      0
diagonal        0
height_left     0
height_right    0
margin_low      0
margin_up       0
length          0
dtype: int64

billets.shape

(1463, 7)

Une concaténation de dataframe 'billets' avec 'test_data' qui contient les valeurs qui ont remplacé les valeurs manquantes

frames=[billets,test_data]
bill=pd.concat(frames)
bill.shape

(1500, 7)

bill.isna().sum()

is_genuine      0
diagonal        0
height_left     0
height_right    0
margin_low      0
margin_up       0
length          0
dtype: int64

Maintenant que nos données sont prêtes, nous pouvons passer à la modélisation
2. Modelisation
2.1 Algorithme k-NN avec Scikit-Learn (model de Classification)

-> Commencons par l'Algorithme k-NN
Préparation des variables pour le modèle

# on divise notre tableau on X et Y
x=bill.drop("is_genuine",axis=1)
x.shape # la taille de data set X

(1500, 6)

# on sépare la colonne 'is_genuine'
y=bill["is_genuine"]
y.shape # # la taille de data set Y

(1500,)

# Séparez training / testing set a l'aide de 'train_test_split'
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8) # le modèle prend 80% pour train set et 20% pour test

Création et Entraînement du modèle de Classification

# import 'Neighbors' depuit sklearn 
from sklearn import neighbors

# Initialiser notre model.  
knn = neighbors.KNeighborsClassifier(n_neighbors=4) # Selectionner un estimateur
knn.fit(xtrain, ytrain) # entrainer le modèle

KNeighborsClassifier(n_neighbors=4)

    l'algorithme ici n'effectue aucune optimisation mais va juste sauvegarder toutes les données en mémoire. C'est sa manière d'apprendre en quelque sorte.

# On pourrait utiliser la methode predict qui obtient des prédictions sur les données de test
knn.predict(xtest)
print(knn.predict(xtest))
print(knn.predict_proba(xtest)) # calculer la probabilité, si les billets ont les caractéristiques False ou True

[ True False  True False False  True  True  True  True  True  True  True
  True  True False  True  True  True  True  True  True  True  True  True
 False False  True False  True  True False  True False  True False  True
 False  True False  True  True  True  True  True  True False  True  True
  True  True False  True  True  True False  True  True  True  True  True
  True False False  True  True False  True False False  True  True  True
  True False  True  True  True  True  True  True False  True  True False
  True False False  True  True False False False  True  True  True  True
  True False False  True False False  True  True  True  True False False
  True  True False False  True  True  True  True False  True  True  True
  True  True  True  True False  True  True  True  True  True  True  True
  True  True False  True  True  True False  True  True False False  True
  True  True  True False False  True  True False  True False  True  True
  True  True  True False False False  True  True  True  True False  True
 False  True False  True False False  True False False  True  True False
  True  True False False  True  True  True False  True  True  True False
  True  True  True  True  True  True  True False False  True  True  True
  True  True False  True  True  True  True False  True  True  True  True
  True  True  True  True  True  True  True  True  True  True False  True
  True  True False False  True False  True False False False  True False
 False  True  True False False  True  True  True False  True False  True
 False False False  True False False  True False False  True False  True
  True False  True False  True False  True  True False  True  True False
  True False  True  True False  True False  True  True False False False
  True False False False  True  True  True  True  True  True  True  True]
[[0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.25 0.75]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.75 0.25]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.25 0.75]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.25 0.75]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [1.   0.  ]
 [1.   0.  ]
 [1.   0.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]
 [0.   1.  ]]

Évaluation du modèle

# Testons à présent l’erreur de notre classifieur
error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)

Erreur: 0.003333

# évaluer les performances de notre modèle
knn.score(xtrain, ytrain) 

0.9908333333333333

    Le résultat est très bon déjà !

-> La performance exacte de notre modèle dépend du jeu de sélection. On peut lancer plusieurs fois ce traitement en changeant les jeux de données d'apprentissage et de test .
Optimisation du score sur les données test

-> Le k (nombre de voisins) est l'hyper-paramètre que l’on va chercher à optimiser, pour minimiser l’erreur sur les données test.

-> Pour trouver le k optimal, on va simplement tester le modèle pour tous les k de 2 à 15, mesurer l’erreur de test et afficher la performance en fonction de k :

# Optimisation du score sur les données test
errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,15), errors, 'o-')
plt.show()

    Pour notre cas, le nombre de voisins 4 est optimal et c'est ce que j'ai choisi pour initialiser le modèle

Verification d'algoritme avec nouveau dataset ' billets_production' pour voir s'il fonctionne bien

bi = pd.read_csv('billets_production.csv')
bi.head()

	diagonal 	height_left 	height_right 	margin_low 	margin_up 	length 	id
0 	171.76 	104.01 	103.54 	5.21 	3.30 	111.42 	A_1
1 	171.87 	104.17 	104.13 	6.00 	3.31 	112.09 	A_2
2 	172.00 	104.58 	104.29 	4.99 	3.39 	111.57 	A_3
3 	172.49 	104.55 	104.34 	4.44 	3.03 	113.20 	A_4
4 	171.65 	103.63 	103.56 	3.77 	3.16 	113.33 	A_5

i=bi.drop("id",axis=1)
i.shape

(5, 6)

# convertir nouveau dataset en 'test'
xtest=i

# utiliser la methode predict
print(knn.predict(xtest))
print(knn.predict_proba(xtest))

[False False False  True  True]
[[1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [0. 1.]]

    Le Resultat de Teste: les 3 premiers tickets sont FAUX, les 2 suivants sont VRAI

                                     ******************

Maintenant on peut envisager une autre méthode de modélisation
2.2 La Régression Logictique avec Scikit-Learn

-> Puisqu'il faut expliquer les variable qualitative à partir de variables explicatives quantitatives, nous pouvons appliquer ici un type de modélisation:

    la Classification supervisée ( ou régression logistique)

-> Ci-dessous voyons comment nous allons tester notre modèle avec la régression loqistique :
2.2.1 Préparation des variables pour le modèle

# import  Regression Logistic depuit sklearn 
from sklearn.linear_model import LogisticRegression

# créer un modèle Regression Logistique 
LogisticRegression(C=1.0,class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1,l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty=12,
                   random_state=None, solver='lbfgs', tol=0.0001,
                   verbose=0, warm_start=False)
lr = LogisticRegression()

# nous divisons notre tableau en X et Y
x1=bill.drop("is_genuine",axis=1) # on sépare la colonne 'is_genuine'
y1=bill["is_genuine"]

# la taille de data set X
x1.shape

(1500, 6)

# la taille de data set Y
y1.shape

(1500,)

# Séparez training / testing set 
x1train, x1test, y1train, y1test = train_test_split(x1, y1, train_size=0.8)

2.2.2 Entraînement du modèle

# fit data
lr.fit(x1train,y1train)

LogisticRegression()

2.2.3 Appliquation le modèle

# faire des prédictions sur les données de test
prediction=lr.predict(x1test)
prediction

array([False,  True, False,  True,  True,  True,  True,  True,  True,
        True, False,  True,  True, False,  True, False,  True,  True,
        True,  True,  True,  True,  True, False, False,  True, False,
        True,  True, False, False,  True,  True,  True, False, False,
        True,  True, False,  True, False,  True,  True,  True,  True,
       False, False,  True,  True,  True, False, False,  True,  True,
        True,  True, False,  True,  True,  True,  True,  True,  True,
       False,  True, False,  True,  True,  True,  True, False, False,
        True, False,  True, False,  True,  True, False,  True,  True,
        True,  True,  True,  True,  True,  True, False,  True,  True,
        True,  True, False,  True,  True, False,  True,  True, False,
        True, False,  True,  True,  True,  True, False,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
       False,  True,  True,  True, False, False, False,  True,  True,
        True,  True, False,  True,  True,  True,  True,  True,  True,
       False, False,  True, False,  True, False, False,  True, False,
        True,  True,  True,  True,  True,  True, False, False,  True,
       False,  True, False, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False,  True,  True,  True,  True,
        True,  True,  True,  True,  True, False,  True, False,  True,
        True,  True,  True, False,  True,  True, False,  True,  True,
       False,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True, False, False, False,  True,
        True,  True,  True, False, False, False, False,  True, False,
        True,  True, False,  True, False,  True,  True,  True,  True,
       False,  True, False,  True, False,  True, False, False,  True,
        True,  True,  True,  True, False,  True, False,  True, False,
        True,  True, False,  True, False,  True, False,  True, False,
        True, False,  True, False,  True, False, False,  True,  True,
       False,  True,  True, False,  True, False,  True,  True, False,
        True,  True, False,  True,  True,  True, False,  True,  True,
        True,  True,  True,  True,  True,  True,  True, False,  True,
        True,  True, False,  True,  True,  True,  True, False, False,
        True,  True,  True])

2.2.4 Évaluation les modèle avec des classification métriques 'Accuracy ' et 'F-score'
! C'est une étape importante dans tout machine learning modele, qui consiste à évaluer la précision du modèle.
Les métriques : 'Accuracy ' et 'F-score' et Les métriques : Mean Squared Error, Mean absolute error, Root Mean Squared Error sont utilisées pour évaluer les performances du modèle dans l'analyse de régression.

-> L’accuracy est basée sur la matrice de confusion et répond à la question : combien d’individus ont été correctement prédits. Elle mesure le taux de prédictions correctes sur l’ensemble des individus

from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
print("Metric: Accuracy score: {:.4f}".format(accuracy_score(prediction,y1test)))
print('Metric: F-score', fbeta_score(prediction,y1test, average=None, beta=2))

Metric: Accuracy score: 0.9867
Metric: F-score [0.98468271 0.98753595]

    Nous avons ici La justesse est 0.99 (soit 99 prédictions correctes sur 100 exemples au total). Cela signifie que notre modéle est très efficace pour identifier les vrais et faux billets.

    -> Mais cependant cette métrique doit être interprétée avec prudence en présence de données déséquilibrées . Donc on explore d’autres métriques de classification 'F-score' pour confirmer le résultat.

    -> F-score métriques accorde la même importance à la precision et au recall. Le taux de précision est un indice qui quantifie la quantité prevue correcte . Il est calculé en divisant le nombre d'exemples positifs prévus correct par le nombre total d'exemples positifs prévus.

Recall Est une mesure du nombre de prédictions positives correctes à partir de toutes les prédictions positives possibles .

    Dans notre cas il sont [ 0.99（ Précision complète ou parfaite ）et 0.99 （ complet ou parfait ）]

2.2.5 Évaluation de la performence du modèle avec 'Mean_absolute_error' et 'Mean_squared_error' metriques

    J'ai continue l'Évaluation de la perforemence de notre modèle avec le mesure des erreurs à l'aide des métriques MAE et MSE.

-> MAE (l'erreur absolue moyennere) représente la moyenne de la différence absolue entre les valeurs réelles et prédites dans l'ensemble de données. Il mesure la moyenne des résidus.

MSE (L'erreur quadratique moyenne) représente la moyenne de la différence au carré entre les valeurs d'origine et prédites. Il mesure la variance des résidus.

RMSE (Racine de l'erreur quadratique moyenne) est la racine carrée de l'erreur quadratique moyenne. Il mesure l'écart type des résidus.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# évaluation du modèle pour l'ensemble d'entraînement
y1test=y1test.astype(np.float32)
prediction=prediction.astype(np.float32)
mae=mean_absolute_error(y_true=y1test,y_pred=prediction)

# squared True renvoie la valeur MSE , Fals renvoie la valeur RMSE
mse= (mean_squared_error(y_true=y1test, y_pred=prediction)) #defaut =True
rmse = (mean_squared_error(y_true=y1test, y_pred=prediction, squared=False))

print("La performance du Modèle pour le set de Training")
print("------------------------------------------------")
print("Metric: [MAE: {:.4f}]".format(mae))
print("Metric: [MSE: {:.4f}]".format(mse))
print("Metric: [RMSE: {:.4f}]".format(rmse))

La performance du Modèle pour le set de Training
------------------------------------------------
Metric: [MAE: 0.0133]
Metric: [MSE: 0.0133]
Metric: [RMSE: 0.1155]

-> Le MAE et le RMSE peuvent être utilisés ensemble pour diagnostiquer la variation des erreurs.

-> Le RMSE sera toujours supérieur ou égal au MAE ; plus la différence entre eux est grande, plus la variance des erreurs individuelles est grande.

    Le MAE et le RMSE peuvent aller de 0 à l'infini. Ce sont des scores orientés négativement : les valeurs inférieures sont meilleures. Dans notre cas, ce diagnostic donne un très bon résultat.

Pour l'évaluation optimale du modèle, il est possible avoir une analyse des nombres de faux positifs et faux négatifs via une Matrice de confusion.

-> La matrice de confusion est la base principale sur laquelle reposent toutes les métriques de classification : précision, score F1, courbe ROC, Précision-Rappel... Ainsi, la maîtriser est un pré-requis important pour comprendre comment évaluer les performances d'un modèle de classification.

-> C' est un outil de mesure de la performance des modèles de classification à 2 classes ou plus. Dans le cas binaire, la matrice de confusion est un tableau à 4 valeurs représentant les différentes combinaisons de valeurs réelles et valeurs prédites.

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y1test, prediction)
print(matrix)

[[ 90   3]
 [  1 206]]

    Notre Martice calcule les prédictions correct - 90 (vrais négatifs) et 206 (vrais positifs) et les prédictions erronées 3 (faux négatifs) et 1 (faux positifs). On peut dire que notre modèle est parfait.
    La matrice de confusion dépend du seuil de classification! Le code a un seuil de classification implicite (au niveau de 50%).

# Verification d'algoritme avec nouveau dataset ' billets_production'
x1test=i
print(lr.predict(x1test))

[False False False  True  True]

    Le Resultat de Teste: les 3 premiers billets sont FALSE, les 2 suivants sont TRUE. C'est le meme que dans la méthode précédente.

    Les deux méthodes sont bien capable d’identifier automatiquement les vrais des faux billets, mais je préfère retenir la seconde comme modèle final en raison d'une évaluation plus solide avec les metriques appliquées.

new=pd.read_csv('billets_test.csv')
new

	diagonal 	height_left 	height_right 	margin_low 	margin_up 	length 	id
0 	172.09 	103.95 	103.73 	4.39 	3.09 	113.19 	B_1
1 	171.52 	104.17 	104.03 	5.27 	3.16 	111.82 	B_2
2 	171.78 	103.80 	103.75 	3.81 	3.24 	113.39 	B_3
3 	172.02 	104.08 	103.99 	5.57 	3.30 	111.10 	B_4
4 	171.79 	104.34 	104.37 	5.00 	3.07 	111.87 	B_5

new1=new.drop('id', axis=1)
new1.shape

(5, 6)

x1test=new1

print(lr.predict(x1test))

[ True False  True False False]

print(lr.predict_proba(x1test))

[[1.03096295e-02 9.89690371e-01]
 [9.92351756e-01 7.64824354e-03]
 [8.21854268e-04 9.99178146e-01]
 [9.99851604e-01 1.48395520e-04]
 [9.87921315e-01 1.20786847e-02]]

 

