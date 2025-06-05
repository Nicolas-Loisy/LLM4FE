import logging
from typing import Dict, Any
from openai import OpenAI as OpenAIClient
from openai.types.responses import Response, ParsedResponse

from src.llm.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)

class OpenAI(BaseLLM):

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize OpenAI LLM.
        
        Args:
            api_key: OpenAI API key
            model: The GPT model to use (default: gpt-4o)
        """
        self._api_key = api_key
        self._model = model
        self._client = OpenAIClient(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response: Response = self._client.responses.create(
                model=self._model,
                input=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.output_text
        except Exception as e:
            logger.error(f"Erreur d'appel OpenAI : {e}")
            return ""

    def generate_with_format(self, prompt: str, response_format, **kwargs) -> Dict[str, Any]:
        try:
            response: ParsedResponse = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                **kwargs
            )
            return response.choices[0].message.parsed

        except Exception as e:
            logger.error(f"Erreur d'appel OpenAI : {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self._model,
            "api_key": "*****" + self._api_key[-3:]
        }
