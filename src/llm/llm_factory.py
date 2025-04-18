from pydantic import BaseModel


class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]


class LLMFactory:

    @staticmethod
    def create_llm(config: dict):
        import requests
        import json
        provider = config["provider"]
        if provider == "Pleiade":
            from llms.pleiade import Pleiade

            api_url = Pleiade.get_api_url()
            api_key = Pleiade.get_api_key()

            Pleiade.set_model(config["model"])
            model = Pleiade.get_model()
            prompt = config.get("prompt")

            request_body = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "format": Country.model_json_schema()
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            response = requests.post(
                api_url,
                headers=headers,
                data=json.dumps(request_body)
            )
            json_formatted = json.dumps(response.json(), indent=4)
            print(json_formatted)

        elif provider == "OpenAI":  # Notez le 'elif' ici au lieu d'un nouveau 'if'
            from llama_index.llms.openai import OpenAI
            from llms.openai import OpenAI
            api_url = OpenAI.get_api_url()
            api_key = OpenAI.get_api_key()
            model = OpenAI.get_model()

            return OpenAI(
                model=model,
                api_key=api_key
            )
        else:
            raise ValueError(f"LLM provider {provider} is not supported.")


config = {
    "provider": "Pleiade",
    "model": "llama3.3:latest",
    "prompt": "Tell me about Canada."
}

reponse = LLMFactory.create_llm(config)


# CIBLE
"""             return Pleiade(
                model=config["model"],
            ) """

# CIBLE
config = {
    "provider": "Pleiade",
    "model": "llama3.3:latest",
}


# CIBLE DANS FE


class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]
