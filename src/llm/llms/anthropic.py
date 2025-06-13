import logging
from typing import Dict, Any
import anthropic
from anthropic import Anthropic as AnthropicClient

from src.llm.llms.base_llm import BaseLLM
from src.models.dataset_model import DatasetStructure

logger = logging.getLogger(__name__)

class Anthropic(BaseLLM):

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Anthropic LLM.
        
        Args:
            api_key: Anthropic API key
            model: The Claude model to use (default: claude-3-5-sonnet-20241022)
        """
        self._api_key = api_key
        self._model = model
        self._client = AnthropicClient(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=kwargs.get('max_tokens', 4096),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Erreur d'appel Anthropic : {e}")
            return ""

    def generate_with_format(self, prompt: str, response_format, **kwargs) -> DatasetStructure:
        try:
            # For structured output, we append format instructions to the prompt
            structured_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema: {response_format}"
            
            response = self._client.messages.create(
                model=self._model,
                max_tokens=kwargs.get('max_tokens', 4096),
                messages=[{"role": "user", "content": structured_prompt}],
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            
            import json
            response_data = json.loads(response.content[0].text)
            logger.debug(f"Response data keys: {list(response_data.keys())}")
            logger.debug(f"Full response data: {response_data}")
            
            # Transform the response data to match DatasetStructure field names
            transformed_data = {
                "datasetDescription": response_data.get("dataset_description", ""),
                "datasetStructure": response_data.get("new_columns", [])
            }
            
            return DatasetStructure(**transformed_data)
            
        except Exception as e:
            logger.error(f"Erreur d'appel Anthropic : {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self._model,
            "api_key": "*****" + self._api_key[-3:]
        }
