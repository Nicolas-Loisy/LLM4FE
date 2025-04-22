from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import requests
from openai import OpenAI as OpenAIClient


class DatasetStructureItem(BaseModel):
    new_column_name: str
    source_columns: List[str]
    column_description: str
    transformation_type: str
    # Peut être précisé selon le type attendu
    transformation_params: Dict[str, Any]


class ResponseFormat(BaseModel):
    dataset_structure: List[DatasetStructureItem]


class OpenAI:
    _available_models = [
        "dall-e-3", "dall-e-2", "gpt-4o-audio-preview-2024-10-01", "babbage-002",
        "tts-1-hd-1106", "text-embedding-3-large", "text-embedding-ada-002", "tts-1-hd",
        "gpt-4o-mini-audio-preview", "gpt-4o-audio-preview", "o1-preview-2024-09-12",
        "gpt-3.5-turbo-instruct-0914", "gpt-4o-mini-search-preview", "tts-1-1106",
        "davinci-002", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo",
        "gpt-4o-mini-search-preview-2025-03-11", "gpt-4o-2024-11-20", "whisper-1",
        "gpt-4o-2024-05-13", "gpt-3.5-turbo-16k", "o1-preview", "gpt-4o-search-preview",
        "gpt-4.5-preview", "gpt-4.5-preview-2025-02-27", "gpt-4o-search-preview-2025-03-11",
        "tts-1", "omni-moderation-2024-09-26", "text-embedding-3-small", "gpt-4o-mini-tts",
        "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4o-transcribe", "gpt-4o-mini-2024-07-18",
        "gpt-4o-mini-transcribe", "o1-mini", "gpt-4o-mini-audio-preview-2024-12-17", "gpt-3.5-turbo-0125",
        "o1-mini-2024-09-12", "omni-moderation-latest"
    ]

    _model = "gpt-4o-mini"
   # _api_key A AJOUTER
    _api_url = "https://api.openai.com/v1/chat/completions"
    _client = OpenAIClient(api_key=_api_key)

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
        cls._client = OpenAIClient(api_key=key)

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
        try:
            completion = cls._client.chat.completions.create(
                model=cls.get_model(),
                messages=[{"role": "user", "content": prompt}]
            )
            content = completion.choices[0].message.content

            if response_format:
                try:
                    parsed = response_format.model_validate_json(content)
                    return parsed
                except Exception as e:
                    print(f"Erreur de parsing du modèle Pydantic : {e}")
                    print(f"Contenu brut : {content}")
                    return None

            return content

        except Exception as e:
            print(f"Erreur d’appel OpenAI : {e}")
            return None


prompt = """
Tu es expert en feature engineering. 
Les colonnes : taille, poids, âge, masse, sexe. 
Crée de nouvelles colonnes pertinentes pour détecter l'obésité.

Données :
[
  { "taille": 175, "poids": 70, "âge": 28, "masse": 22.9, "sexe": "H" },
  { "taille": 162, "poids": 60, "âge": 34, "masse": 22.9, "sexe": "F" },
  { "taille": 180, "poids": 85, "âge": 45, "masse": 26.2, "sexe": "H" }
]
"""

# Sans parsing
print(OpenAI.call(prompt))

# Avec parsing
print(OpenAI.call(prompt, response_format=ResponseFormat))
