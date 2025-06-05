import logging
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from src.feature_engineering.transformation_factory import TransformationFactory
from src.feature_engineering.transformations.base_transformation import BaseTransformation
from src.llm.llm_factory import LLMFactory
from src.models.dataset_model import DatasetStructure, Transformation
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    def __init__(self, dataset_path: str, dataset_description: Optional[str] = None, target_column: Optional[str] = None):
        """
        Initialize Feature Engineering Pipeline.
        
        Args:
            dataset_path: Path to the input dataset (CSV file).
            dataset_description: Optional description of the dataset to guide transformations.
            target_column: Optional target column for supervised learning tasks.
        """
        self.dataset_path: Path = Path(dataset_path)
        self.dataset_description: Optional[str] = dataset_description
        self.target_column: Optional[str] = target_column
        self.transformations: List[Transformation] = []
        self.input_dataset: Optional[pd.DataFrame] = None
        self.transformed_dataset: Optional[pd.DataFrame] = None
        self.config = get_config()
        
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
        transforms_text = self.get_available_transformations_info()
        target_info = self.get_target_column_info()

        prompt = self.config.get_with_params("prompt_file", {
            "dataset_info": dataset_info,
            "dataset_description": self.dataset_description or 'No description provided',
            "target_info": target_info,
            "transforms_text": transforms_text
        })
        
        if not prompt:
            logger.error("Failed to load prompt template")
            return [], None
            
        logger.debug("Loaded prompt template successfully")
        logger.debug(f"Prompt: {prompt}")
        
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
                self.dataset_description = response.datasetDescription
                logger.info(f"Dataset description: {self.dataset_description}")
                logger.info(f"Generated {len(self.transformations)} transformations.")
                return self.transformations, self.dataset_description
            else:
                logger.error(f"Unexpected response format from LLM: {response}")
                return [], None

        except Exception as e:
            logger.error(f"Error generating transformations: {e}")
            return [], None

    def get_target_column_info(self) -> str:
        """
        Get information about the target column if specified.
        
        Returns:
            String with target column information.
        """
        # Add target column information to the prompt if specified
        if self.target_column:
            target_info = f"\nTarget column for prediction: {self.target_column}"
            
            # Add more specific details about the target if available
            if self.input_dataset is not None and self.target_column in self.input_dataset.columns:
                target_data = self.input_dataset[self.target_column]
                target_type = target_data.dtype
                unique_values = target_data.nunique()
                
                target_info += f"\nTarget column type: {target_type}"
                target_info += f"\nNumber of unique values: {unique_values}"
                
                # For categorical target, show value distribution
                if unique_values <= 10:
                    value_counts = target_data.value_counts(normalize=True)
                    target_info += "\nClass distribution:"
                    for val, count in value_counts.items():
                        target_info += f"\n  - {val}: {count:.1%}"
            return target_info
        else:
            return "No target column specified."

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

    def get_available_transformations_info(self) -> str:
        """
        Get information about available transformations.
        
        Returns:
            String with available transformations information.
        """
        logger.info("Getting available transformations information...")
        available_transforms = []
        info_transforms = TransformationFactory.INFO_TRANSFORMATIONS
        
        for provider, description in info_transforms.items():
            available_transforms.append(f"- {provider}: {description}")
        transforms_text = "Available transformations with descriptions:\n" + "\n".join(available_transforms)
        
        return transforms_text

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
                transformation: BaseTransformation = TransformationFactory.create_transformation(transform_config.model_dump())
                
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

    def run(self) -> Tuple[pd.DataFrame, List[Transformation], Optional[str]]:
        """
        Main entry point to run a single feature engineering iteration.
        
        Returns:
            Tuple containing:
            - DataFrame with the transformed dataset
            - List of transformations applied
            - Optional dataset description
        """
        logger.info("Starting Feature Engineering pipeline for a single iteration...")
        
        # Generate transformations
        transformations, description = self.generate_transformations()
        
        # Set the transformations
        self.transformations = transformations
        
        # Apply transformations
        transformed_dataset = self.apply_transformations()
        
        if transformed_dataset is not None:
            logger.info("Feature engineering iteration completed successfully")
            self.transformed_dataset = transformed_dataset
        else:
            logger.error("Feature engineering iteration failed to produce transformed dataset")
        
        return self.transformed_dataset, self.transformations, description
