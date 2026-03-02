<h1> DESSAUX_Damien_ECF3 </h1>

ECF3 de la formation *Développeur Concepteur en Science des Donnée* de M2i (JEDHA 35288).

# 1. Table of Contents
 
- [1. Table of Contents](#1-table-of-contents)
- [2. Description du projet](#2-description-du-projet)
- [3. Desscription du dataset](#3-desscription-du-dataset)
  - [3.1. Description des varaibles explicatives:](#31-description-des-varaibles-explicatives)
  - [3.2. Description de la variable cible](#32-description-de-la-variable-cible)
- [4. Structure du projet](#4-structure-du-projet)
- [5. Prérequis](#5-prérequis)
- [6. Installation](#6-installation)
  - [6.1. Cloner le projet depuis GitHub.](#61-cloner-le-projet-depuis-github)
  - [6.2. Créer un environement virtuel et installer les dépendances.](#62-créer-un-environement-virtuel-et-installer-les-dépendances)
  - [6.3. Démarrer l'infrastructure Docker.](#63-démarrer-linfrastructure-docker)
- [7. Utilisation](#7-utilisation)
  - [7.1. Scripts Jupyter](#71-scripts-jupyter)
  - [7.2. Pipelines Spark Mllib](#72-pipelines-spark-mllib)

# 2. Description du projet

L'objectif du projet est de construire un modèle prédictif pour identifier les clients à risque de départ (`Churn`).

L'étude suivra les étapes suivantes :
1. Analyse exploratoire des données (EDA)
2. Prétraitement & Feature Engineering
3. Comparaisons et évaluations de modèles
4. Optimisation du meilleur modèle
5. Implémentation distribuée avec Spark MLlib
6. Comparaison Scikit-Learn et Spark MLlib

# 3. Desscription du dataset

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

# 4. Structure du projet

```
DESSAUX_DAMIEN_ECF3/
├── .gitignore                  # Fichiers et dossiers ignorés par Git
├── docker-compose.yaml         # Orchestration des services (Docker)
├── Dockerfile                  # Image Docker pour Spark
├── pyproject.toml              # Configuration du projet Python (PEP 518)
├── README.md                   # Documentation principale
├── requirements.txt            # Dépendances Python
│
├── data/
│   ├── 03_DONNEES.csv          # Dataset utilisé pour l'analyse
│   └── enriched_dataset.csv    # Dataset enrichie lors de la phase de feature engineering
│
├── docs/
│   ├── rapport.md              # Bilan (méthodologie, résultats et recommandations)
│   └── SUJET_ECF3.md           # Sujet du projet
│
├── logs/
│   └── 05_spark_mllib.log      # Logs d'exécution Spark MLlib
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Analyse exploratoire des données
│   ├── 02_preprocessing.ipynb  # Génération des pipelines de prétraitement
│   ├── 03_modelisation.ipynb   # Comparaisons et évaluations de modèles
│   ├── 04_optimisation.ipynb   # Optimisation du meilleur modèle
│   ├── 05_spark_mllib.py       # Implémentation Spark MLlib
│   └── 06_scikit_learn_spark_comparison.ipynb  # Comparaison Scikit-Learn et Spark MLlib
│
└── output/
    ├── figures/                # Visualisations générées
    │
    ├── metrics/                # Métriques calculées
    │
    ├── models/                 # Pipelines de prétraitement et Modèles entraînés
    │
    └── predictions_test.csv    # Résultats de prédiction
```

# 5. Prérequis

- Docker et Docker Compose
- Python 3.13+
- Git

# 6. Installation

## 6.1. Cloner le projet depuis GitHub.

```bash
git clone https://github.com/DamienDESSAUX-M2i/DESSAUX_Damien_ECF3.git
```

## 6.2. Créer un environement virtuel et installer les dépendances.

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
## Linux/Mac:
source venv/bin/activate
## Windows:
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

Vous pouvez également utilisez `uv` avec la commande `uv sync`.

## 6.3. Démarrer l'infrastructure Docker.

Avant de démarrer l'infrastructure, veillez à ce que les dossier `notebooks`, `data`, `output` et `logs` soient créés.

Quatre services seront lancés `spark-master`, `spark-worker-1`, `spark-worker-2` et `spark-worker-3`.
Ces services sont construis à partir du fichier `DorkerFile` qui ajoute à l'image `apache/spark:3.5.3` les bibliothèques `pandas` et `numpy`.

```bash
docker-compose up -d
```

# 7. Utilisation

## 7.1. Scripts Jupyter

Les fichiers `.ipynb` ne sont pas indépendants et doivent être lancé dans l'ordre numérique.
Ces fichiers seront exécutés via `Jupyter lab`.

```bash
jupyter lab
```

## 7.2. Pipelines Spark Mllib

Pour lancer la pipeline de machine learning `05_spark_mllib` utilisez la commande :

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /notebooks/05_spark_mllib.py
```
