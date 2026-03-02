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
  - [5.1. EDA](#51-eda)
  - [5.2. Comparison des modèles](#52-comparison-des-modèles)
  - [5.3. Optimisation](#53-optimisation)
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

Description des features ajoutées :
| Feature | Description |
| :- | :- |
| **NumberServices** | Nombre de services additionnels activés par client. |
| **AverageMonthly** | Dépense mensuelle moyenne (`TotalCharges / tenure`, fallback sur `MonthlyCharges`). |
| **TenureSegment** | Segment d’ancienneté : novice (≤6), adepte (> 6 et ≤24), confirmé (>24). |
| **ChargePerService** | Coût mensuel divisé par le nombre de services (`MonthlyCharges / (NumberServices + 1)`). |

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

Le modèle ayant obtenue le meilleur recall est optimisé et évalué de nouveau. La métrique utilisée pour l'optimisation est le *F1 score* afin d'améliorer la *precision* du modèle sans sacrifier le *recall*.

## 4.5. Spark MLlib

Plusieurs modèles sont entrainés et évalués. La métrique reste la même.
Tous les modèles utilisent un prétraitement naïf et certaines variantes des modèles utilisent la pondération (`weightCol="weight"`) pour gérer le déséquilibre des classes.

| Pipeline | Modèle | Gestion des classes |
| :- | :- | :- |
| `lr` | LogisticRegression | Standard |
| `lr_balanced` | LogisticRegression | Balanced(`weightCol="weight"`) |
| `rfc` | RandomForest | Standard |
| `rfc_balanced` | RandomForest | Balanced(`weightCol="weight"`) |
| `gbc` | GradientBoostedTrees | Standard |
| `gbc_balanced` | GradientBoostedTrees | Balanced |

## 4.6. Comparaison Scikit-learn

Les deux approches de machine learning Scikit-learn et Spark MLlib sont comparées. On s'interesse notament aux métriques *recall*, *F1 score* et au temps d'exécution.

# 5. Résultats

## 5.1. EDA

L'étude des variables explicatives a révélé :
- une distribution relativement équilibrée des variables catégorielles ormis pour les variables de services internet pour lesquelles la modalité *No service internet* est deux fois plus importante que les autres modalités ([Distribution variables catégorielles](../output/figures/01_categorical_features_distribution.png)).
- des corrélations entre les services internet, notamment entre la variable *InternetService* avec les services internet ([Corrélations variables catégorielles](../output/figures/01_correlation_matrix_cat.png)).
- une distribution uniforme de la varible *tenure* et une distribution asymétrique avec une queue de série à gauche lourde pour les trois autres variables numériques ([Distribution variables numériques](../output/figures/01_numeric_features_analyses.png)).
- la possibilité de transformer la variable *TotalCharges* pour la symétriser ([Transformation TotalCharges](../output/figures/01_transformation_TotalCharges.png)).
- des corrélation entre les variables numériques, notamment entre la variable *MonthlyCharges* et *InternetCharges*, montrant une redondance d'information ([Corrélations variables numériques](../output/figures/01_correlation_matrix_num.png)).
- des corrélation entre les services internet et les variables numériques, en particulier entre *InternetService* et *MonthlyCharges* ou *InternetCharges*, montrant une redondance d'information ([Corrélations variables numériques et catégorielles](../output/figures/01_correlation_matrix_cat_num.png)).

L'analyse de la variable cible a montré :
- un déséquilibre des classes ([Distribution Churn](../output/figures/01_churn_distribution.png)).
- une corrélation avec la variable *Contrat*. Les contrats *Month-to-month* ont une proportion de Churn supérieur à la moyenne du dataset et les contrats *Two year* une proportion inférieure ([Distribution varialbes catégorielles en fonction du Churn](../output/figures/01_target_analysis_categorical_features.png), [Corrélation Churn variables catégorielles](../output/figures/01_correlation_matrix_target_cat.png)).
- une corrélation avec la variable *tenure*. La distribution des tenure en fonction du Churn montre a une queue de distribution pour lourde à gauche pour les clients partis et une queue de distribution pour lourde à droite pour les clients restés ([Distributon variables numériques en fonction du Churn](../output/figures/01_target_analyse_numeric_features.png)).

## 5.2. Comparison des modèles

Pour Scikit-Learn,
- le modèle ayant le meilleur *recall* est *LogisticRegression* ([Métrics](../output/metrics/model_comparisons.csv)).
- le prétraitement "feature engineering", comparé au prétraitement "naïf" augmente très légérement le *recall*, mais cette amélioration est négligeable, environ 0,003.
- la modèle *LogisticRegression* capture le Churn mais au détriment de la *precision*. Ce modèle produit beaucoup de faux positifs ([Matrice de confusion](../output/figures/03_confusion_matrix.png)).
- le modèle *LogisticRegression* est sujet au sous-apprentissage ([Courbe d'apprentissage](../output/figures/03_learning_curve.png)), ce qui sugère que les modèles comparés ne sont pas assez complexes.
- l'étude des features importances du modèle *LogistiqueRegression* révèle que la varible *Contract* traité comme ordinale et la variable *TotalCharges* transformée sont les features les plus importantes ([Métrics](../output/figures/03_feature_importance.png)).

Pour Spark Mllib,
- le modèle ayant le meilleur *recall* est *RandomForestClassifier* ([Métrics](../output/metrics/model_comparison_spark.csv)).
- l'étude des features importances montre que les varialbes *Contract*, *tenure* et *TotalCharges* ont le plus d'influence ([Features importance spark](../logs/05_spark_mllib.log)).

## 5.3. Optimisation

Les paramètres optimisés sont : *C*, *solver* et *fit_intercept*.
L'optimisation du modèle *LogisticRegression* n'a pas permis d'amèliorer significativement ses performances ([Matrice de confusion modèle optimisé](../output/figures/04_confusion_matrix.png)).

# 6. Recommandations

On analyse le comportement des 20 clients du jeu de test ayant la plus forte probabilité prédite par le modèle *LogisticRegression* de partir ([Analyse clients partis](../output/figures/03_top20_churn_analysis.png)).

On observe que ces clients :
- ont pour la grande majorité des contrat *Month-to-month*.
- ont pour la grande majorité la fibre optique.
- sont de nouveaux clients avec des contrat d'une durée de vie inférieur ou égale à 6 mois.
- ont leur montant médian des charges total proche de 139€, ce qui est supérieur à la médiane du dataset qui est de 91€. 

Les résultats précédents montrent que les variables *Contract*, *tenure*, *TotalCharges* et *MonthlyCharges* sont les plus importantes.

En conclusion nous recommandons une action sur :
- la durée des contrats : inciter les clients à opter pour des contrats à plus longue durée.
- le montant des charges totales et charges par mois : veillez à ce que les clients ne s'écartent pas trop de la médiane.
- cibler l'action sur les adérents de moins de 6 mois.