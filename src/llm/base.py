from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLM(ABC):
    """
    Abstract base class for Large Language Models.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: The input prompt for the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    def generate_with_format(self, prompt: str, response_format: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Generate text based on the given prompt with specified response format.
        
        Args:
            prompt: The input prompt for the model
            response_format: Optional format specification for the response (e.g., {"type": "json"})
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated response in the specified format
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            A dictionary containing model information
        """
        pass
