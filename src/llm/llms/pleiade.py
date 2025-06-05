import logging
import requests
import json
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel

from src.llm.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)

class Pleiade(BaseLLM):
    """
    Pleiade implementation of the LLM interface.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama3.3:latest"):
        """
        Initialize Pleiade LLM.
        
        Args:
            api_key: Optional API key for authentication
            model: The model to use
        """
        self._api_key = api_key
        self._model = model
        self._api_url = "https://pleiade.mi.parisdescartes.fr/api/chat/completions"

    def generate(self, prompt: str, response_format: Optional[Type[BaseModel]] = None):
        request_body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}]
        }

        if response_format:
            request_body["format"] = response_format.model_json_schema()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        response = requests.post(
            self._api_url,
            headers=headers,
            data=json.dumps(request_body)
        )

        try:
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur : {e}")
            return None
        
    def generate_with_format(self, prompt: str, response_format: Optional[Type[BaseModel]] = None):
        """
        Generate text using Pleiade with specified format.
        
        Args:
            prompt: The input prompt
            response_format: Format specification for the response
            
        Returns:
            The generated response in the specified format
        """
        return self.generate(prompt, response_format)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self._model,
            "api_key": "*****" + self._api_key[-3:]
        }