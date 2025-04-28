from src.llm.llm_factory import LLMFactory
from src.models.dataset_model import DatasetStructure


# config = {
#     "provider": "OpenAi",
#     "api_key": "",
#     "model": "gpt-4o",
# }

config = {
    "provider": "OpenWebUI",
    "api_key": "",
    "api_url": "https://pleiade.mi.parisdescartes.fr",
    "model": "llama3.3:latest",
}

# config = {
#     "provider": "Pleiade",
#     "api_key": "",
#     "api_url": "https://pleiade.mi.parisdescartes.fr/api/chat/completions",
#     "model": "llama3.3:latest",
# }


prompt = """
Je veux créer un dataset pour un modèle de machine learning.

Voici les colonnes attendues :
- "age": âge de la personne, extrait de la colonne "birth_date" via un calcul de différence de date.
- "salary_euros": conversion du salaire initial (colonne "salary_usd") en euros.
- "is_adult": booléen indiquant si la personne a plus de 18 ans.

Merci de me donner la description sous forme du modèle demandé.
"""

llm = LLMFactory.create_llm(config)
# response = llm.generate("Hello, world!")
# print(response)
response = llm.generate_with_format(prompt=prompt, response_format=DatasetStructure)
print(response)
print(type(response))

# TODO : Ajouter model des transformations
# TODO : Transformer le json return par le LLM en objet DatasetStructure