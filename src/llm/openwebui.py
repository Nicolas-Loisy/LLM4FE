import requests
import json
from typing import Dict, Any, Optional
from .base import LLM


class OpenWebUILLM(LLM):
    """
    OpenWebUI implementation of the LLM interface.
    """
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize OpenWebUI LLM.
        
        Args:
            api_url: The URL of the OpenWebUI API
            api_key: Optional API key for authentication
        """
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenWebUI.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the OpenWebUI API
            
        Returns:
            The generated text response
        """
        payload = {
            "prompt": prompt,
            **kwargs
        }
        
        response = requests.post(
            f"{self.api_url}/api/generate",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"OpenWebUI API error: {response.status_code} - {response.text}")
    
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
        # Create messages array for compatibility with OpenWebUI
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": kwargs.pop("model", "default"),
            "messages": messages,
            **kwargs
        }
        
        # Add response_format to the payload directly if specified
        if response_format:
            # OpenWebUI expects the format specification at the top level
            if "type" in response_format:
                if response_format["type"] == "json":
                    payload["format"] = {"type": "json"}
                elif response_format["type"] == "json_schema":
                    payload["format"] = response_format
                else:
                    payload["format"] = response_format
            else:
                payload["format"] = response_format
        
        response = requests.post(
            f"{self.api_url}/api/chat/completions",  # Using chat endpoint for format support
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            result = response_data.get("response", "")
            
            # If the API already parsed the JSON, use that
            if "parsed_output" in response_data:
                return response_data["parsed_output"]
                
            # Otherwise try to parse JSON if that was the requested format
            if response_format and (response_format.get("type") == "json" or response_format.get("type") == "json_schema"):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # If parsing fails, return the raw string
                    return result
            
            return result
        else:
            raise Exception(f"OpenWebUI API error: {response.status_code} - {response.text}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current OpenWebUI model.
        
        Returns:
            A dictionary containing model information
        """
        response = requests.get(
            f"{self.api_url}/api/models/info",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get model info: {response.status_code}"}
