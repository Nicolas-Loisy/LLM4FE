import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import json
import datetime

from src.feature_engineering.transformation_factory import TransformationFactory
from src.feature_engineering.transformations.base_transformation import BaseTransformation
from src.llm.llm_factory import LLMFactory
from src.models.dataset_model import DatasetStructure, Transformation
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    def __init__(self, dataset_path: str, dataset_description: Optional[str] = None):
        """
        Initialize Feature Engineering Pipeline.
        
        Args:
            dataset_path: Path to the input dataset (CSV file).
            dataset_description: Optional description of the dataset to guide transformations.
        """
        self.dataset_path: Path = Path(dataset_path)
        self.dataset_description: Optional[str] = dataset_description
        self.transformations: List[Transformation] = []
        self.input_dataset: Optional[pd.DataFrame] = None
        self.transformed_dataset: Optional[pd.DataFrame] = None
        self.config = get_config()
        self.version = 0
        self.version_history = []
        self.output_dir = self.dataset_path.parent / "versions"
        self.output_dir.mkdir(exist_ok=True)
        
        try:
            self.llm = LLMFactory.create_llm(self.config.get("llm"))
        except Exception as e:
            logger.error(f"Error initialising LLM: {e}")
            self.llm = None
             
        if not self.load_dataset():
            raise ValueError(f"Failed to load dataset from {self.dataset_path}")

    def load_dataset(self) -> bool:
        """
        Load the dataset from a CSV file.
        
        Args:
            dataset_path: Path to the input dataset (CSV file).
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"File not found: {self.dataset_path}")
            
            self.input_dataset = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset successfully loaded: {self.input_dataset.shape}")
            logger.debug(f"Dataset loaded: {self.input_dataset.head()}")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def set_transformations(self, transformations: List[Transformation]):
        """
        Set transformations manually.
        
        Args:
            transformations: List of transformation configurations.
        """
        self.transformations = transformations
        logger.info(f"Transformations set manually: {len(self.transformations)} transformations.")

    def generate_transformations(self) -> Tuple[List[Transformation], Optional[str]]:
        """
        Generate transformations using LLM.
            
        Returns:
            Tuple[List[Transformation], Optional[str]]: List of transformation configurations and optional dataset description.
        """
        logger.info("Generating transformations...")
        
        dataset_info = self.get_dataset_info()

        available_transforms = []
        info_transforms = TransformationFactory.INFO_TRANSFORMATIONS
        for provider, description in info_transforms.items():
            available_transforms.append(f"- {provider}: {description}")
        transforms_text = "Available transformations with descriptions:\n" + "\n".join(available_transforms)
        
        prompt = f"""
        You are a data scientist tasked with creating feature engineering transformations for a machine learning model.
                
        Here's information about the dataset:
        {dataset_info}

        Dataset description: {self.dataset_description or 'No description provided'}

        {transforms_text}

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
            logger.info("Generating transformations with LLM...")
            response: DatasetStructure = self.llm.generate_with_format(
                prompt=prompt,
                response_format=DatasetStructure
            )
            logger.info("LLM response received.")
            logger.debug(f"LLM response: {response}")

            # Extract the transformations from the response
            if isinstance(response, DatasetStructure):
                self.transformations = response.datasetStructure
                self.dataset_description = response.dataset_description
                logger.info(f"Dataset description: {self.dataset_description}")
                logger.info(f"Generated {len(self.transformations)} transformations.")
                return self.transformations, self.dataset_description
            else:
                logger.error(f"Unexpected response format from LLM: {response}")
                return [], None

        except Exception as e:
            logger.error(f"Error generating transformations: {e}")
            return [], None

    def get_dataset_info(self) -> str:
        """
        Get informations about the dataset for the LLM prompt.
            
        Returns:
            String with dataset information.
        """
        logger.info("Getting dataset information...")
        
        dataset_info = []
        dataset_info.append(f"Dataset shape: {self.input_dataset.shape}")
        
        dataset_info.append("\nColumn information:")
        for col in self.input_dataset.columns:
            dtype = self.input_dataset[col].dtype
            unique_count = self.input_dataset[col].nunique()
            null_count = self.input_dataset[col].isna().sum()
            null_percentage = (null_count / len(self.input_dataset)) * 100
            
            dataset_info.append(
                f"- {col}: type={dtype}, unique_values={unique_count}, "
                f"null_values={null_count} ({null_percentage:.1f}%)")
            
            # Add sample values for categorical columns
            if dtype == 'object' and unique_count <= 10:
                sample_values = self.input_dataset[col].dropna().unique()[:5]
                dataset_info.append(f" Sample values: {', '.join(str(v) for v in sample_values)}")
        
        numeric_cols = self.input_dataset.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            dataset_info.append("\nNumeric columns summary:")
            stats = self.input_dataset[numeric_cols].describe().to_string()
            dataset_info.append(stats)
        
        return "\n".join(dataset_info)

    def apply_transformations(self) -> pd.DataFrame:
        """
        Apply transformations to the dataset.
        
        Returns:
            Transformed dataframe.
        """
        logger.info("Applying transformations...")
        
        if self.input_dataset is None:
            logger.error("No dataset loaded. Please load a dataset first.")
            return None
        
        if not self.transformations:
            logger.warning("No transformations to apply.")
            self.transformed_dataset = self.input_dataset.copy()
            return self.transformed_dataset
        
        self.transformed_dataset = self.input_dataset.copy()
        
        for transform_config in self.transformations:
            try:
                transformation: BaseTransformation = TransformationFactory.create_transformation(transform_config)
                
                if transformation:
                    self.transformed_dataset = transformation.transform(self.transformed_dataset)
                    logger.info(f"Applied transformation: {transformation}")
                else:
                    logger.error(f"Failed to create transformation: {transform_config}")
            except Exception as e:
                logger.error(f"Error applying transformation {transform_config}: {e}")
                continue

        logger.info(f"Applied {len(self.transformations)} transformations.")
        logger.info(f"Transformed dataset shape: {self.transformed_dataset.shape}")
        
        return self.transformed_dataset

    def save_versioned_dataset(self, dataset: pd.DataFrame, version: int) -> Path:
        """
        Save a versioned dataset to CSV.
        
        Args:
            dataset: DataFrame to save
            version: Version number
            
        Returns:
            Path to the saved dataset
        """
        filename = f"{self.dataset_path.stem}_v{version}.csv"
        output_path = self.output_dir / filename
        dataset.to_csv(output_path, index=False)
        logger.info(f"Saved version {version} to {output_path}")
        return output_path

    def save_fe_configuration(self, version: int, transformations: List[Transformation], dataset_description: str, input_path: Path, output_path: Path) -> Path:
        """
        Save fe configuration information for a version.
        
        Args:
            version: Version number
            transformations: Applied transformations
            dataset_description: Dataset description
            input_path: Path to input dataset
            output_path: Path to output dataset
            
        Returns:
            Path to the saved fe configuration file
        """
        config_filename = f"{self.dataset_path.stem}_config_v{version}.json"
        config_path = self.output_dir / config_filename
        
        # Convert transformations directly to JSON-serializable format
        json_transformations = []
        for t in transformations:
            json_transformations.append(t.model_dump())

        # Current version data
        current_version_data = {
            "version": version,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_description": dataset_description,
            "input_file": str(input_path),
            "output_file": str(output_path),
            "dataset_shape": {
                "rows": self.transformed_dataset.shape[0],
                "columns": self.transformed_dataset.shape[1]
            },
            "transformations": json_transformations,
            "columns": list(self.transformed_dataset.columns)
        }
        
        # Full fe configuration with version history
        config_data = {
            "current_version": current_version_data,
            "version_history": self.version_history  # Include all previous versions
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved fe configuration for version {version} to {config_path}")
        return config_path

    def run(self, iterations: int = 1) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Main entry point to run the complete feature engineering pipeline with versioning support.
        
        Args:
            iterations: Number of transformation iterations to perform
            
        Returns:
            Tuple containing:
            - DataFrame with the final transformed dataset
            - List of version history dictionaries with metadata for each version
        """
        logger.info(f"Starting Feature Engineering pipeline with {iterations} iterations...")
        
        current_dataset = self.input_dataset.copy()
        input_path = self.dataset_path
        
        for i in range(iterations):
            self.version += 1
            logger.info(f"Starting iteration {i+1}/{iterations} (version {self.version})...")
            
            # Set current dataset as the input
            self.input_dataset = current_dataset
            
            # Generate transformations for this iteration
            transformations, description = self.generate_transformations()
            
            # Apply transformations
            current_dataset = self.apply_transformations()
            
            if current_dataset is not None:
                # Save the transformed dataset
                output_path = self.save_versioned_dataset(current_dataset, self.version)
                
                # Update history before saving fe configuration
                version_entry = {
                    "version": self.version,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "transformations_count": len(transformations),
                    "description": description,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                self.version_history.append(version_entry)
                
                # Save fe configuration with all history
                config_path = self.save_fe_configuration(
                    self.version, 
                    transformations, 
                    description or self.dataset_description or "", 
                    input_path, 
                    output_path
                )
                
                # Add config path to history entry
                self.version_history[-1]["config_path"] = str(config_path)
                
                logger.info(f"Iteration {i+1} completed successfully, created version {self.version}")
                
                # Update for next iteration
                input_path = output_path
            else:
                logger.error(f"Iteration {i+1} failed to produce transformed dataset.")
                break
        
        self.transformed_dataset = current_dataset
        logger.info(f"Pipeline completed with {self.version} versions created.")
        
        return self.transformed_dataset, self.version_history
