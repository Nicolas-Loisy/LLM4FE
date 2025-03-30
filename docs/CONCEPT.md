# LLM4FE

### **1. Objectif du projet**

Développer un **pipeline automatisé** de **Feature Engineering (FE)**, **AutoML**, et **Benchmarking**, orchestré par un **Orchestrateur central**.  
Le projet doit permettre de **transformer dynamiquement un dataset** en utilisant un **LLM** qui génère un JSON de transformations via **response format / structured output**, puis d'entraîner un modèle AutoML et de benchmarker ses performances, en stockant les résultats dans une structure versionnée.

---

## **2. Structure du projet**

Le projet suit une structure modulaire et organisée :

```
/LLM4FE
│
├── 📂 data/  # Contient les datasets et modèles sauvegardés
│   ├── models/  # Sauvegarde des modèles entraînés
│   ├── logs/  # Fichiers de log de l'exécution du pipeline
│
├── 📂 src/
│   ├── __init__.py
│
│   ├── 📂 orchestrator/  # Module central de gestion du pipeline
│   │   ├── __init__.py
│   │   ├── orchestrator.py  # Coordination du pipeline
│   │   ├── config.py  # Gestion des configurations
│
│   ├── 📂 feature_engineering/  # Module de transformation des données
│   │   ├── __init__.py
│   │   ├── fe_pipeline.py  # Exécution des transformations
│   │   ├── fe_factory.py  # Factory pour gérer dynamiquement les transformations
│   │
│   │   ├── 📂 transformations/  # Dossier contenant les transformations spécifiques
│   │   │   ├── __init__.py
│   │   │   ├── base_transform.py  # Classe parent pour toutes les transformations
│   │   │   ├── scaling.py  # Transformations de mise à l'échelle
│   │   │   ├── encoding.py  # Encodage des variables catégorielles
│   │   │   ├── text_processing.py  # Traitement du texte (TF-IDF, embeddings, etc.)
│   │   │   ├── math_operations.py  # Fonctions mathématiques (log, moyenne, etc.)
│
│   ├── 📂 llm/  # Gestion des appels au LLM
│   │   ├── __init__.py
│   │   ├── llm_factory.py  # Factory pour supporter plusieurs modèles LLM
│   │   ├── openwebui_client.py  # Client pour l’API OpenWebUI
│
│   ├── 📂 automl/  # Entraînement des modèles
│   │   ├── __init__.py
│   │   ├── automl_pipeline.py  # Exécution du processus AutoML
│
│   ├── 📂 benchmark/  # Évaluation des modèles
│   │   ├── __init__.py
│   │   ├── benchmark.py  # Calcul des scores des modèles
│
├── requirements.txt
├── README.md
├── .gitignore
└── run_pipeline.py  # Script principal lançant le pipeline
```

---

## **3. Description détaillée des modules et interactions**

### **3.1. Orchestrateur (`src/orchestrator/`)**

L'Orchestrateur **coordonne les interactions entre les modules** et gère les versions des datasets et modèles.

#### **Responsabilités :**

- **Charger les fichiers d’entrée** (`dataset.csv`, `config.json`, etc.).
- **Gérer les itérations et versions** :
  - Générer `version_1`, `version_2`, etc.
  - Sauvegarder chaque version sous `/data/models/version_X/`
- **Appeler le module Feature Engineering** et récupérer le **dataset transformé**.
- **Appeler le module AutoML** et récupérer le **modèle entraîné**.
- **Appeler le module Benchmarking** et récupérer les **scores du modèle**.
- **Sélectionner la meilleure version** en comparant les scores.

---

### **3.2. Feature Engineering (`src/feature_engineering/`)**

Le module FE applique les transformations sur le dataset en **appelant un LLM** via l'API OpenWebUI.

#### **Responsabilités :**

- **Générer un JSON structuré** via un **appel LLM** utilisant un **response format / structured output**.
- **Appliquer dynamiquement** les transformations via un **FE Factory**.
- **Sauvegarder le dataset transformé** (`Dataset_FE_vX.csv`).

#### **Appel au LLM avec response format**

Le JSON généré par le LLM suit cette **structure stricte** :

```python
from pydantic import BaseModel
from typing import List, Optional, Literal

class Transformation(BaseModel):
    finalCol: str
    colToProcess: List[str]
    providerTransform: Literal['math', 'aggregation', 'encoding', 'scaling', 'custom']
    param: Optional[str]

class DatasetStructure(BaseModel):
    datasetStructure: List[Transformation]
```

#### **Exemple d’appel au LLM :**

```python
import requests
import json

url = "https://openwebui.example/api/chat/completions"
api_key = "API_KEY"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

prompt = "Génère des transformations adaptées à ce dataset..."

data = {
    "model": "llama3.3:latest",
    "messages": [{"role": "user", "content": prompt}],
    "format": DatasetStructure.model_json_schema()
}

response = requests.post(url, headers=headers, data=json.dumps(data))
transformations = response.json()
```

---

### **3.3. AutoML (`src/automl/`)**

Le module **entraîne automatiquement des modèles** en utilisant des bibliothèques comme **Auto-sklearn** ou **TPOT**.

#### **Responsabilités :**

- Charger le dataset transformé (`Dataset_FE_vX.csv`).
- Sélectionner un algorithme (Random Forest, XGBoost, etc.).
- Entraîner le modèle.
- Sauvegarder le modèle (`model_vX.pkl`).

---

### **3.4. Benchmarking (`src/benchmark/`)**

Le module **évalue les performances** du modèle entraîné.

#### **Responsabilités :**

- Charger le modèle (`model_vX.pkl`).
- Calculer les scores (`accuracy`, `F1-score`, `AUC`, etc.).
- Sauvegarder les scores (`Model_Scores_vX.json`).
- Comparer les scores des différentes versions.

---

## **4. Processus d’itération**

1️ **L’Orchestrateur démarre l’itération 1** :

- Il envoie le dataset brut au **module FE**.

2️ **Le module Feature Engineering** :

- Génère un **JSON structuré** avec un **LLM (response format)**.
- Applique les transformations et sauvegarde `Dataset_FE_v1.csv`.

3️ **Le module AutoML** :

- Charge `Dataset_FE_v1.csv`.
- Entraîne un modèle et sauvegarde `model_v1.pkl`.

4️ **Le module Benchmarking** :

- Évalue `model_v1.pkl` et sauvegarde `Model_Scores_v1.json`.

5️ **L’Orchestrateur passe à l’itération 2** :

- Il relance le **Feature Engineering** avec `Dataset_FE_v1.csv`.
- Le cycle recommence jusqu’à obtenir `Dataset_FE_vN.csv`.

6️ **Sélection de la meilleure version** :

- L’Orchestrateur compare les scores et sélectionne le **meilleur modèle et dataset**.

---

## **5. Technologies utilisées**

- **Python**
- **OpenWebUI API** pour les appels LLM
- **Pydantic** pour valider les JSON de transformations
- **Scikit-learn, Auto-sklearn, TPOT** pour AutoML
- **Pandas, NumPy** pour la manipulation des données
