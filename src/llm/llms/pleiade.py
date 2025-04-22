from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import requests
import json


class DatasetStructureItem(BaseModel):
    new_column_name: str
    source_columns: List[str]
    column_description: str
    transformation_type: str
    # Peut être précisé selon le type attendu
    transformation_params: Dict[str, Any]


class ResponseFormat(BaseModel):
    dataset_structure: List[DatasetStructureItem]


class Pleiade:
    _available_models = [
        "athene-v2:latest",
        "llama3.3:latest",
        "llama3.2-vision:latest",
        "llama3.2-vision:90b",
        "llava:34b",
        "mathstral:latest",
        "mistral-small:latest",
        "qwen2-math:latest",
        "qwen2.5:latest",
        "qwq:latest",
        "tulu3:70b",
        "yi-coder:latest",
        "codellama:7b-code",
        "deepseek-R1:latest",
        "minicpm-v:latest",
        "phi4-mini:latest",
        "qwen2-math:72b"
    ]

    _model = "llama3.3:latest"
   #  _api_key = A AJOUTER
    _api_url = "https://pleiade.mi.parisdescartes.fr/api/chat/completions"

    @classmethod
    def set_model(cls, model: str):
        if model not in cls._available_models:
            raise ValueError(
                f"Le modèle '{model}' n'est pas disponible sur Pleiade.")
        cls._model = model

    @classmethod
    def get_model(cls):
        return cls._model

    @classmethod
    def set_api_key(cls, key: str):
        cls._api_key = key

    @classmethod
    def get_api_key(cls):
        if not cls._api_key:
            raise ValueError("La clé API Pleiade n’a pas été définie.")
        return cls._api_key

    @classmethod
    def get_api_url(cls):
        return cls._api_url

    @classmethod
    def call(cls, prompt: str, response_format: Optional[Type[BaseModel]] = None):
        request_body = {
            "model": cls.get_model(),
            "messages": [{"role": "user", "content": prompt}]
        }

        if response_format:
            request_body["format"] = response_format.model_json_schema()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cls.get_api_key()}"
        }

        response = requests.post(
            cls.get_api_url(),
            headers=headers,
            data=json.dumps(request_body)
        )

        try:
            response.raise_for_status()
            response_json = response.json()
            print(json.dumps(response_json, indent=4))
            return response_json
        except requests.exceptions.RequestException as e:
            print(f"Erreur : {e}")
            print(response.text)
            return None


# voir branch de yvonne pour les co avec les transfo
# nommage branche feature/LLM num ticket
config = {
    "provider": "Pleiade",
    "model": "llama3.3:latest",
    "prompt": """
    Tu es expert en feature engineering et voici un ensemble de données sous format JSON. 
    Les colonnes disponibles sont : taille, poids, âge, masse, sexe. 
    À partir de ces données, ta tâche est de créer de nouvelles colonnrd pertinentes. Pour savoir si les personnes sont atteintes d'obésité.
    Donne des exemples de transformations, de calculs ou d'agrégations sur ces données.

    Données : 
    [
      { "taille": 175, "poids": 70, "âge": 28, "masse": 22.9, "sexe": "H" },
      { "taille": 162, "poids": 60, "âge": 34, "masse": 22.9, "sexe": "F" },
      { "taille": 180, "poids": 85, "âge": 45, "masse": 26.2, "sexe": "H" }
    ]
    """
}

"""
Pleiade.call(config["prompt"])
Pleiade.call(config["prompt"], response_format=ResponseFormat)"""
