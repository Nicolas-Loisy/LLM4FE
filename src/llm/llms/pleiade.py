class Pleiade:
    _available_models = [
        "athene-v2:latest",
        "llama3.3:latest",
        "llama3.2-vision:latest",
        "llama3.2-vision:90b",
        "llava:34b",
        "mathstral:latest",
        "mistral-small:latest",
        "qwen2-math:latest",
        "qwen2.5:latest",
        "qwq:latest",
        "tulu3:70b",
        "yi-coder:latest",
        "codellama:7b-code",
        "deepseek-R1:latest",
        "minicpm-v:latest",
        "phi4-mini:latest",
        "qwen2-math:72b"
    ]

   # mettre un model par défaut  _model =
    _api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjcxMTEwNzM3LTA1ZmMtNDhlNS05MDg5LTllOTI0MjFkZDFmZiJ9.07UkYLz-0rTcwIG9Rkwr8Ql7_JqoFaO3OtUSKiMDkJ0"
    _api_url = "https://pleiade.mi.parisdescartes.fr/api/chat/completions"

    @classmethod
    def set_model(cls, model: str):
        if model not in cls._available_models:
            raise ValueError(
                f"Le modèle '{model}' n'est pas disponible sur Pleiade.")
        cls._model = model

    @classmethod
    def get_model(cls):
        return cls._model

    @classmethod
    def set_api_key(cls, key: str):
        cls._api_key = key

    @classmethod
    def get_api_key(cls):
        if not cls._api_key:
            raise ValueError("La clé API Pleiade n’a pas été définie.")
        return cls._api_key

    @classmethod
    def get_api_url(cls):
        return cls._api_url


# faire une fonction call avec les attributs call( prompt, response format)
# Avec cette archi , c'est simple de tester plusieurs llm pour le FE
