import requests
import json
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel

from src.llm.llms.base_llm import BaseLLM


class OpenWebUILLM(BaseLLM):
    """
    OpenWebUI implementation of the LLM interface.
    """
    
    def __init__(self, api_url: str, api_key: Optional[str] = None, model: str = "llama3.3:latest"):
        """
        Initialize OpenWebUI LLM.
        
        Args:
            api_url: The URL of the OpenWebUI API
            api_key: Optional API key for authentication
            model: The model to use (default: llama3.3:latest)
        """
        self._api_url = api_url
        self._api_key = api_key
        self._model = model
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
    
    def generate(self, prompt: str, response_format: Optional[Type[BaseModel]] = None, **kwargs) -> Any:
        request_body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}]
        }

        if response_format:
            request_body["format"] = response_format.model_json_schema()

        response = requests.post(
            f"{self._api_url}/api/chat/completions",
            headers=self._headers,
            data=json.dumps(request_body)
        )
        
        try:
            response.raise_for_status()
            response_json = response.json()
            if response_format:
                return response_format.model_validate_json(response_json.get("choices", [{}])[0].get("message", {}).get("content", {}))
            return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"Erreur : {e}")
            return None
    
    def generate_with_format(self, prompt: str, response_format: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Generate text using OpenWebUI with specified format.
        
        Args:
            prompt: The input prompt
            response_format: Format specification for the response
            **kwargs: Additional parameters for the OpenWebUI API
            
        Returns:
            The generated response in the specified format
        """
        return self.generate(prompt, response_format)
        
 
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current OpenWebUI model.
        
        Returns:
            A dictionary containing model information
        """
        response = requests.get(
            f"{self._api_url}/api/models/info",
            headers=self._headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get model info: {response.status_code}"}
