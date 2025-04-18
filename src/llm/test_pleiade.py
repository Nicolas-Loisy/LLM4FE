import requests
import json
from pydantic import BaseModel

# Définir le schéma de sortie structuré


class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]


# URL du serveur Ollama
url = "https://pleiade.mi.parisdescartes.fr/api/chat/completions"

# Clé API (si nécessaire)
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjcxMTEwNzM3LTA1ZmMtNDhlNS05MDg5LTllOTI0MjFkZDFmZiJ9.Up_Yj4GpX9gF0yCAQZIt2d2K0QYLHhYrH3TFxphPSww"

# En-têtes de la requête
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Données de la requête avec schéma structuré
data = {
    "model": "llama3.3:latest",
    "messages": [{"role": "user", "content": "Tell me about Canada."}],
    "format": Country.model_json_schema()  # Passer le schéma en tant que JSON
}

# Faire la requête
response = requests.post(url, headers=headers, data=json.dumps(data))

# Convertir en JSON avec indentation
json_formatted = json.dumps(response.json(), indent=4)

# Afficher le JSON formaté
print(json_formatted)
