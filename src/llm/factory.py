from typing import Dict, Any, Optional
from .base import LLM
from .openwebui import OpenWebUILLM
from .gpt import GPTLLM


class LLMFactory:
    """
    Factory class for creating LLM instances.
    """
    
    @staticmethod
    def create_llm(llm_type: str, config: Dict[str, Any]) -> LLM:
        """
        Create an LLM instance based on the specified type and configuration.
        
        Args:
            llm_type: The type of LLM to create ("openwebui" or "gpt")
            config: Configuration dictionary for the LLM
            
        Returns:
            An instance of the specified LLM
            
        Raises:
            ValueError: If an unsupported LLM type is provided
        """
        if llm_type.lower() == "openwebui":
            return OpenWebUILLM(
                api_url=config.get("api_url"),
                api_key=config.get("api_key")
            )
        elif llm_type.lower() == "gpt":
            return GPTLLM(
                api_key=config.get("api_key"),
                model=config.get("model", "gpt-3.5-turbo")
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
