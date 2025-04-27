from typing import List, Dict, Any
from pydantic import BaseModel
from src.llm import OpenWebUILLM, Pleiade, OpenAI


# TODO : Les classes de structure de données doivent être définies dans un autre fichier
class ColumnStructure(BaseModel):
    new_column_name: str
    column_description: str
    source_columns: List[str]
    transformation_type: str
    # TODO : Précisé selon le type de transformation attendue
    transformation_params: Dict[str, Any]

class DatasetStructure(BaseModel):
    dataset_structure: List[ColumnStructure]


class LLMFactory:

    @staticmethod
    def create_llm(llm_config: dict):
        provider = llm_config["provider"]
        if provider == "Pleiade":
            return Pleiade(api_key=llm_config["api_key"], model=llm_config["model"]) 

        if provider == "OpenAi":
            return OpenAI(api_key=llm_config["api_key"], model=llm_config.get("model"))

        if provider == "OpenWebUI":
            return OpenWebUILLM(api_url=llm_config["api_url"], api_key=llm_config["api_key"], model=llm_config["model"])
        
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

# reponse = LLMFactory.create_llm(llm_config)

# print(reponse)
