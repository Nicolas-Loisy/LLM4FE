import requests
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from typing import List, Optional, Literal

from src.feature_engineering.fe_factory import FeatureEngineeringFactory


class Transformation(BaseModel):
    finalCol: str
    colToProcess: List[str]
    providerTransform: Literal['math', 'aggregation', 'encoding', 'scaling', 'text', 'custom']
    param: Optional[str] = None


class DatasetStructure(BaseModel):
    datasetStructure: List[Transformation]


class FeatureEngineeringPipeline:
    def __init__(self):
        self.factory = FeatureEngineeringFactory()
        self.transformations = []
        self.input_dataset = None
        self.transformed_dataset = None
        self.version = 1
        self.api_key = os.environ.get("OPENWEBUI_API_KEY", "API_KEY")
        self.api_url = os.environ.get("OPENWEBUI_API_URL", "https://openwebui.example/api/chat/completions")
        self.model = os.environ.get("LLM_MODEL", "llama3.3:latest")

    def load_dataset(self, dataset_path: str) -> bool:
        """
        Load the dataset from a CSV file.
        
        Args:
            dataset_path: Path to the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.input_dataset = pd.read_csv(dataset_path)
            print(f"Loaded dataset with shape: {self.input_dataset.shape}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def set_version(self, version: int):
        """Set the version number for this pipeline run."""
        self.version = version

    def generate_transformations(self, dataset_description: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate transformations using LLM.
        
        Args:
            dataset_description: Optional description of the dataset
            
        Returns:
            List of transformation configurations
        """
        print("Generating transformations...")
        
        if self.input_dataset is None:
            print("No dataset loaded. Please load a dataset first.")
            return []
        
        # Prepare the prompt for the LLM
        dataset_info = self._get_dataset_info()
        
        prompt = f"""
        You are a data scientist tasked with creating feature engineering transformations for a machine learning model.
        
        Here's information about the dataset:
        {dataset_info}
        
        {dataset_description or ''}
        
        Generate a list of feature engineering transformations that would improve model performance.
        For each transformation, specify:
        1. The final column name (finalCol)
        2. The column(s) to process (colToProcess)
        3. The type of transformation (providerTransform): 'math', 'aggregation', 'encoding', 'scaling', 'text', or 'custom'
        4. Any parameters needed (param)
        
        Return the transformations in a structured JSON format.
        """
        
        # Prepare the request to the LLM API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": DatasetStructure.model_json_schema()
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the response
            transformations_data = response.json()
            
            # Extract the transformations from the response
            if isinstance(transformations_data, dict) and 'datasetStructure' in transformations_data:
                self.transformations = transformations_data['datasetStructure']
            else:
                print("Unexpected response format from LLM")
                self.transformations = []
            
            print(f"Generated {len(self.transformations)} transformations")
            return self.transformations
            
        except Exception as e:
            print(f"Error generating transformations: {e}")
            # Fallback to some default transformations for testing
            self._generate_default_transformations()
            return self.transformations

    def _generate_default_transformations(self):
        """Generate some default transformations for testing."""
        self.transformations = []
        
        if self.input_dataset is None:
            return
        
        # Add some basic transformations based on column types
        numeric_cols = self.input_dataset.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.input_dataset.select_dtypes(include=['object']).columns.tolist()
        
        # Add scaling for numeric columns
        for col in numeric_cols[:2]:  # Just use the first few columns
            self.transformations.append({
                "finalCol": f"{col}_scaled",
                "colToProcess": [col],
                "providerTransform": "scaling",
                "param": "standard"
            })
        
        # Add encoding for categorical columns
        for col in categorical_cols[:2]:  # Just use the first few columns
            self.transformations.append({
                "finalCol": f"{col}_encoded",
                "colToProcess": [col],
                "providerTransform": "encoding",
                "param": "label"
            })

    def _get_dataset_info(self) -> str:
        """Get information about the dataset for the LLM prompt."""
        if self.input_dataset is None:
            return "No dataset loaded."
        
        info = []
        info.append(f"Dataset shape: {self.input_dataset.shape}")
        
        # Column information
        info.append("\nColumn information:")
        for col in self.input_dataset.columns:
            dtype = self.input_dataset[col].dtype
            unique_count = self.input_dataset[col].nunique()
            null_count = self.input_dataset[col].isna().sum()
            
            info.append(f"- {col}: type={dtype}, unique_values={unique_count}, null_values={null_count}")
            
            # Add sample values for categorical columns
            if self.input_dataset[col].dtype == 'object' and unique_count < 10:
                sample_values = self.input_dataset[col].dropna().unique()[:5]
                info.append(f"  Sample values: {', '.join(str(v) for v in sample_values)}")
        
        # Statistical summary for numeric columns
        numeric_cols = self.input_dataset.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info.append("\nNumeric columns summary:")
            stats = self.input_dataset[numeric_cols].describe().to_string()
            info.append(stats)
        
        return "\n".join(info)

    def apply_transformations(self) -> pd.DataFrame:
        """
        Apply transformations to the dataset.
        
        Returns:
            Transformed dataframe
        """
        print("Applying transformations...")
        
        if self.input_dataset is None:
            print("No dataset loaded. Please load a dataset first.")
            return None
        
        if not self.transformations:
            print("No transformations to apply.")
            self.transformed_dataset = self.input_dataset.copy()
            return self.transformed_dataset
        
        # Create a copy of the input dataset
        self.transformed_dataset = self.input_dataset.copy()
        
        # Apply each transformation
        for transform_config in self.transformations:
            # Create the transformation
            transformation = self.factory.create_transformation(transform_config)
            
            if transformation:
                # Apply the transformation
                self.transformed_dataset = transformation.transform(self.transformed_dataset)
                print(f"Applied transformation: {transformation}")
            else:
                print(f"Failed to create transformation: {transform_config}")
        
        print(f"Applied {len(self.transformations)} transformations")
        print(f"Transformed dataset shape: {self.transformed_dataset.shape}")
        
        return self.transformed_dataset

    def save_transformed_dataset(self, output_dir: str = "data") -> str:
        """
        Save the transformed dataset.
        
        Args:
            output_dir: Directory to save the dataset
            
        Returns:
            Path to the saved dataset
        """
        print("Saving transformed dataset...")
        
        if self.transformed_dataset is None:
            print("No transformed dataset to save.")
            return None
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the dataset
        output_path = os.path.join(output_dir, f"Dataset_FE_v{self.version}.csv")
        self.transformed_dataset.to_csv(output_path, index=False)
        
        print(f"Saved transformed dataset to {output_path}")
        return output_path
