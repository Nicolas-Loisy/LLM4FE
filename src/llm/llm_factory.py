from typing import List, Dict, Any
from pydantic import BaseModel
from llms.pleiade import Pleiade
from llms.openai_client import OpenAI


class DatasetStructureItem(BaseModel):
    new_column_name: str
    source_columns: List[str]
    column_description: str
    transformation_type: str
    # Peut être précisé selon le type attendu
    transformation_params: Dict[str, Any]


class ResponseFormat(BaseModel):
    dataset_id: str
    # dataset_source_file: str
    prompt: str  # Non retourner pas le LLM
    target_column: str
    # configs: config
    dataset_structure: List[DatasetStructureItem]


class LLMFactory:

    @staticmethod
    def create_llm(config: dict):
        provider = config["provider"]
        if provider == "Pleiade":
            Pleiade.set_model(config["model"])
            return Pleiade

        elif provider == "OpenAi":
            OpenAI.set_model(config["model"])
            return OpenAI
        else:
            raise ValueError(f"LLM provider {provider} is not supported.")


# POUR TESTER OPEN AI LA REPONSE DOIT ETRE LA CLASSE
'''config = {
    "provider": "OpenAi",
    "model": "dall-e-3",
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
}'''

# POUR TESTER PLEIADE RENVOIE <class 'llms.pleiade.Pleiade'>
'''config = {
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
}'''

reponse = LLMFactory.create_llm(config)

print(reponse)
