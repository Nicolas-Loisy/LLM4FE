import openai
import json
from typing import Dict, Any, Optional
from ..base import LLM


class GPTLLM(LLM):
    """
    GPT implementation of the LLM interface using OpenAI's API.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize GPT LLM.
        
        Args:
            api_key: OpenAI API key
            model: The GPT model to use (default: gpt-3.5-turbo)
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's GPT.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            The generated text response
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def generate_with_format(self, prompt: str, response_format: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Generate text using OpenAI's GPT with specified format.
        
        Args:
            prompt: The input prompt
            response_format: Format specification for the response
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            The generated response in the specified format
        """
        # OpenAI supports response_format directly for certain models
        api_kwargs = {**kwargs}
        
        # Add response_format to the API call if specified
        if response_format:
            # OpenAI uses a specific format for JSON responses
            if response_format.get("type") == "json":
                api_kwargs["response_format"] = {"type": "json"}
            
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **api_kwargs
        )
        
        result = response.choices[0].message.content
        
        # Parse the response if JSON was requested
        if response_format and response_format.get("type") == "json":
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                # If parsing fails, return the raw string
                return result
                
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current GPT model.
        
        Returns:
            A dictionary containing model information
        """
        return {
            "model": self.model,
            "provider": "OpenAI",
            "type": "GPT"
        }
