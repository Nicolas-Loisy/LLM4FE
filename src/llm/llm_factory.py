from src.llm import OpenWebUILLM, Pleiade, OpenAI
from src.llm.llms.anthropic import Anthropic
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
        
        if provider == "Anthropic":
            return Anthropic(api_key=llm_config["api_key"], model=llm_config.get("model", "claude-3-5-sonnet-20241022"))
        else:
            raise ValueError(f"LLM provider {provider} is not supported.")
