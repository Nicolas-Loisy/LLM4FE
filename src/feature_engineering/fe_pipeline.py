import requests
import json
from src.feature_engineering.fe_factory import FeatureEngineeringFactory

# Feature Engineering Pipeline

class FeatureEngineeringPipeline:
    def __init__(self):
        self.factory = FeatureEngineeringFactory()

    def generate_transformations(self):
        # Generate transformations using LLM
        print("Generating transformations...")
        url = "https://openwebui.example/api/chat/completions"
        api_key = "API_KEY"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        prompt = "Génère des transformations adaptées à ce dataset..."
        data = {
            "model": "llama3.3:latest",
            "messages": [{"role": "user", "content": prompt}],
            "format": "json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        transformations = response.json()
        print(transformations)

    def apply_transformations(self):
        # Apply transformations to the dataset
        print("Applying transformations...")

    def save_transformed_dataset(self):
        # Save the transformed dataset
        print("Saving transformed dataset...")
