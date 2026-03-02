<h1> DESSAUX_Damien_ECF3 </h1>

ECF3 de la formation *Développeur Concepteur en Science des Donnée* de M2i (JEDHA 35288).

# 1. Table of Contents
 
- [1. Table of Contents](#1-table-of-contents)
- [2. Description du projet](#2-description-du-projet)
- [3. Structure du projet](#3-structure-du-projet)
- [4. Prérequis](#4-prérequis)
- [5. Installation](#5-installation)
  - [5.1. Cloner le projet depuis GitHub.](#51-cloner-le-projet-depuis-github)
  - [5.2. Créer un environement virtuel et installer les dépendances.](#52-créer-un-environement-virtuel-et-installer-les-dépendances)
  - [5.3. Démarrer l'infrastructure Docker.](#53-démarrer-linfrastructure-docker)
- [6. Utilisation](#6-utilisation)
  - [6.1. Scripts Jupyter](#61-scripts-jupyter)
  - [6.2. Pipelines Spark Mllib](#62-pipelines-spark-mllib)

# 2. Description du projet

L'objectif du projet est de construire un modèle prédictif pour identifier les clients à risque de départ (`Churn`).

L'étude suivra les étapes suivantes :
1. Analyse exploratoire des données (EDA)
2. Prétraitement & Feature Engineering
3. Comparaisons et évaluations de modèles
4. Optimisation du meilleur modèle
5. Implémentation distribuée avec Spark MLlib
6. Comparaison Scikit-Learn et Spark MLlib

# 3. Structure du projet

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
│   ├── rapport.md              # Bilan (résumé, méthodologie, résultats et recommandations)
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
│   └── 06_scikit_learn_spark_comparison.ipynb  # Comparaison
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

# 4. Prérequis

- Docker et Docker Compose
- Python 3.13+
- Git

# 5. Installation

## 5.1. Cloner le projet depuis GitHub.

```bash
git clone https://github.com/DamienDESSAUX-M2i/DESSAUX_Damien_ECF3.git
```

## 5.2. Créer un environement virtuel et installer les dépendances.

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

## 5.3. Démarrer l'infrastructure Docker.

Avant de démarrer l'infrastructure, veillez à ce que les dossier `notebooks`, `data`, `output` et `logs` soient créés.

Quatre services seront lancés `spark-master`, `spark-worker-1`, `spark-worker-2` et `spark-worker-3`.
Ces services sont construis à partir du fichier `DorkerFile` qui ajoute à l'image `apache/spark:3.5.3` les bibliothèques `pandas` et `numpy`.

```bash
docker-compose up -d
```

# 6. Utilisation

## 6.1. Scripts Jupyter

Les fichiers `.ipynb` ne sont pas indépendants et doivent être lancé dans l'ordre numérique.
Ces fichiers seront exécutés via `Jupyter lab`.

## 6.2. Pipelines Spark Mllib

Pour lancer la pipeline de machine learning `05_spark_mllib` utilisez la commande :

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /notebooks/05_spark_mllib.py
```
