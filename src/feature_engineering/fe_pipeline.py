import requests
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from typing import List, Optional, Literal

from src.feature_engineering.transformation_factory import TransformationFactory
from src.feature_engineering.transformations.base_transformation import BaseTransformation
from src.llm.llm_factory import LLMFactory


class Transformation(BaseModel):
    finalCol: str
    colToProcess: List[str]
    providerTransform: Literal[*TransformationFactory.PROVIDER_TRANSFORMATIONS] # Récupère dynamiquement les transformations disponibles
    param: Optional[Dict[str, Any]] = None


class DatasetStructure(BaseModel):
    datasetStructure: List[Transformation]


class FeatureEngineeringPipeline:
    def __init__(self, dataset_path: str, dataset_description: Optional[str] = None):
        """
        Initialize Feature Engineering Pipeline.
        
        Args:
            dataset_description: Optional description of the dataset to guide transformations
        """
        self.transformations = []
        self.input_dataset = None
        self.dataset_description = dataset_description
        self.transformed_dataset = None
        self.version = 1
        
        # TODO : Les variables d'environnement seront bientôt gérées par le fichier config.py, supprimer quand fichier dispo
        # Initialize LLM using factory
        # api_key = os.environ.get("OPENWEBUI_API_KEY", "API_KEY")
        # api_url = os.environ.get("OPENWEBUI_API_URL", "https://openwebui.example/api")
        # model = os.environ.get("LLM_MODEL", "llama3.3:latest")
        
        # llm_config = {
        #     "api_url": api_url,
        #     "api_key": api_key,
        #     "model": model
        # }
        
        # # Create LLM instance
        # self.llm = LLMFactory.create_llm("openwebui", llm_config)
        self.load_dataset(dataset_path)

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
        1. The new column name (new_column_name)
        2. The source columns (source_columns)
        3. The category: 'math', 'aggregation', 'encoding', 'scaling', 'text', or 'custom'
        4. Any transformation parameters (transformation_params)
        
        Return the transformations in a structured JSON format.
        """
        
        try:
            # Use the LLM with format support
            schema = DatasetStructure.model_json_schema()
            
            response = self.llm.generate_with_format(
                prompt=prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "DatasetStructure",
                        "strict": True,
                        "schema": schema
                    }
                }
            )
            
            # Extract the transformations from the response
            if isinstance(response, dict) and 'datasetStructure' in response:
                self.transformations = response['datasetStructure']
            else:
                print("Unexpected response format from LLM")
                self.transformations = []
            
            print(f"Generated {len(self.transformations)} transformations")
            return self.transformations
            
        except Exception as e:
            print(f"Error generating transformations: {e}")

  
    # TODO : Fonction à ignorer pour le moment, permet de récupérer les informations sur le dataset pour le prompt LLM
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

    def apply_transformations(self, transformations: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Apply transformations to the dataset.
        
        Returns:
            Transformed dataframe
        """
        print("Applying transformations...")
        
        if self.input_dataset is None:
            print("No dataset loaded. Please load a dataset first.")
            return None
        
        if not transformations:
            print("No transformations to apply.")
            self.transformed_dataset = self.input_dataset.copy()
            return self.transformed_dataset
        
        # Create a copy of the input dataset
        self.transformed_dataset = self.input_dataset.copy()
        
        # Apply each transformation
        for transform_config in transformations:
            # Create the transformation
            transformation: BaseTransformation = TransformationFactory.create_transformation(transform_config)
            
            if transformation:
                # Apply the transformation
                self.transformed_dataset = transformation.transform(self.transformed_dataset)
                print(f"Applied transformation: {transformation}")
            else:
                print(f"Failed to create transformation: {transform_config}")
        
        print(f"Applied {len(transformations)} transformations")
        print(f"Transformed dataset shape: {self.transformed_dataset.shape}")
        
        return self.transformed_dataset

    # TODO : Fonction à ignorer pour le moment, permet de sauvegarder le dataset transformé
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

    def run(self) -> Dict[str, Any]:
        """
        Main entry point to run the complete feature engineering pipeline.
        
        Args:
            dataset_path: Path to the input dataset
            output_dir: Directory to save the transformed dataset
            
        Returns:
            Dictionary containing information about the transformations and paths
        """
        print("Starting Feature Engineering pipeline...")
                
        # TODO : Remettre après tests
        # Generate transformations
        # transformations = self.generate_transformations(self.dataset_description)
        transformations = self.transformations
        
        # Apply transformations
        transformed_dataset = self.apply_transformations(transformations)
        
        # Save transformed dataset
        # output_path = self.save_transformed_dataset(output_dir)
        
        return transformed_dataset
