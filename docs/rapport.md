<h1>Rapport</h1>

# 1. Table of Contents
 
- [1. Table of Contents](#1-table-of-contents)
- [2. Description du projet](#2-description-du-projet)
- [3. Description du dataset](#3-description-du-dataset)
  - [3.1. Description des varaibles explicatives:](#31-description-des-varaibles-explicatives)
  - [3.2. Description de la variable cible](#32-description-de-la-variable-cible)
- [4. Méthodo](#4-méthodo)
  - [4.1. EDA](#41-eda)
  - [4.2. Prétraitemet et feature engineering](#42-prétraitemet-et-feature-engineering)
  - [4.3. Modélisation et évaluation de modèles](#43-modélisation-et-évaluation-de-modèles)
  - [4.4. Optimisation](#44-optimisation)
  - [4.5. Spark MLlib](#45-spark-mllib)
  - [4.6. Comparaison Scikit-learn](#46-comparaison-scikit-learn)
- [5. Résultats](#5-résultats)
- [6. Recommandations](#6-recommandations)

# 2. Description du projet

L'objectif du projet est de construire un modèle prédictif pour identifier les clients à risque de départ (`Churn`).

L'étude suis les étapes suivantes :
1. Analyse exploratoire des données (EDA)
2. Prétraitement & Feature Engineering
3. Comparaisons et évaluations de modèles
4. Optimisation du meilleur modèle
5. Implémentation distribuée avec Spark MLlib
6. Comparaison Scikit-Learn et Spark MLlib

Le dataset utilisé comprends 

# 3. Description du dataset

Le dataset utilisé comprends 7043 lignes et 21 variables.

## 3.1. Description des varaibles explicatives:

| Variable | Description |
| :- | :- |
| `gender` | Genre du client (Masculin / Féminin). |
| `SeniorCitizen` | 1 si le client est senior, 0 sinon. |
| `Partner` | Indique si le client a un partenaire. |
| `Dependents` | Indique si le client a des personnes à charge. |
| `tenure` | Durée de l’abonnement en mois. |
| `Contract` | Type de contrat (mensuel, annuel, etc.). |
| `PhoneService` | Présence d’un service téléphonique. |
| `MultipleLines` | Indique si le client a plusieurs lignes téléphoniques. |
| `InternetService` | Type de service Internet (DSL, Fibre, Aucun). |
| `OnlineSecurity` | Souscription à la protection en ligne. |
| `OnlineBackup` | Souscription au service de sauvegarde en ligne. |
| `DeviceProtection` | Souscription à la protection de l’appareil. |
| `TechSupport` | Accès au support technique. |
| `StreamingTV` | Utilisation d’un service de streaming TV. |
| `StreamingMovies` | Utilisation d’un service de streaming de films. |
| `InternetCharges` | Frais mensuels du service Internet. |
| `MonthlyCharges` | Charges mensuelles totales du client. |
| `TotalCharges` | Montant total payé depuis le début de l’abonnement. |

## 3.2. Description de la variable cible

| Variable | Description |
| :- | :- |
| `Churn` | Indique si le client est parti (Yes 14.4% / No 86.6%). |

# 4. Méthodo

## 4.1. EDA

L'analyse exploratoire est découpée en deux parties. La première partie se concentre sur l'analyse des varialbes explicatives, on s'interresse aux distributions des varaibles explicatives puis aux corrélations entre variables explicatives. La seconde partie est axée sur l'étude de la variable cible, on s'interresse aux corrélations entre la varibale cible et les variables explicatives.

## 4.2. Prétraitemet et feature engineering

Deux pipeline de prétraitement sont générées. La première pipeline est naïve, elle comprends toutes les varaibles explicatives. Les varaibles catégorielles sont encodées et les variables numériques sont standardisées.
La seconde pipeline est plus aboutie, elle comprend une sélection des variables explicatives d'après l'EDA, une encodage pour les varialbes ordinales et une transformation pour la variable *TotalCharges*.

## 4.3. Modélisation et évaluation de modèles

Plusieurs modèles sont entrainés et évaluées. La métrique choisie pour l'évaluation est le *recall* puisque l'on cherche à déterminer un maximun de client succeptible de partir quitte à créer des faux possitifs.
le modèle avec le meilleur recall est évalué finement (matrice de confusion, cross validation, feature importance, courbe d'apprentissage, courbe ROC, ...).

Le tableau ci-après résume toutes les combinaisons de modèles et de prétraitements utilisées pour le projet.  
Chaque pipeline applique d’abord un préprocesseur (naïf ou avec features engineering) puis entraîne un classificateur.
La colonne "Gestion des classes" indique si le modèle prend en compte un déséquilibre éventuel des classes.

| Pipeline | Modèle | Prétraitement | Gestion des classes |
| :- | :- | :- | :- |
| `lr_naive` | LogisticRegression | Naïf | Standard |
| `lr_balanced_naive` | LogisticRegression | Naïf | Balanced |
| `rfc_naive` | RandomForest | Naïf | Standard |
| `rfc_balanced_naive` | RandomForest | Naïf | Balanced |
| `gbc_naive` | GradientBoosting | Naïf | Standard |
| `lr_feat_eng` | LogisticRegression | FeatureEngineering | Standard |
| `lr_balanced_feat_eng` | LogisticRegression | FeatureEngineering | Balanced |
| `rfc_feat_eng` | RandomForest | FeatureEngineering | Standard |
| `rfc_balanced_feat_eng` | RandomForest | FeatureEngineering | Balanced |
| `gbc_feat_eng` | GradientBoosting | FeatureEngineering | Standard |

Dans cette section on analyse également les 20 clients du jeu de test ayant la plus haute probabilté, prédite par le meilleur modèle, de partir afin de proposer des recommandation.

## 4.4. Optimisation

Le modèle ayant obtenue le meilleur recall est optimisé et évalué de nouveau.

## 4.5. Spark MLlib

Plusieurs modèles sont entrainés et évalués. La métrique reste la même.
Tous les modèles utilisent un prétraitement naïf et certaines variantes des modèles utilisent la pondération (`weightCol="weight"`) pour gérer le déséquilibre des classes.

| Pipeline | Modèle | Gestion des classes |
| :- | :- | :- | :- |
| `lr` | LogisticRegression | Standard |
| `lr_balanced` | LogisticRegression | Balanced(`weightCol="weight"`) |
| `rfc` | RandomForest | Standard |
| `rfc_balanced` | RandomForest | Balanced(`weightCol="weight"`) |
| `gbc` | GradientBoostedTrees | Standard |
| `gbc_balanced` | GradientBoostedTrees | Balanced |

## 4.6. Comparaison Scikit-learn

Les deux approches de machine learning Scikit-learn et Spark MLlib sont comparées. On s'interesse notament aux métriques *recall*, *F1 score* et au temps d'exécution.

# 5. Résultats

TODO

# 6. Recommandations

TODO