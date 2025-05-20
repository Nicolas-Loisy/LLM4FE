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

    def generate_transformations(self) -> List[Transformation]:
        """
        Generate transformations using LLM.
            
        Returns:
            List of transformation configurations.
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
                logger.info(f"Generated {len(self.transformations)} transformations.")
                return self.transformations
            else:
                logger.error(f"Unexpected response format from LLM: {response}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating transformations: {e}")
            return []

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

    def run(self) -> Tuple[pd.DataFrame, List[Transformation]]:
        """
        Main entry point to run the complete feature engineering pipeline.
        
        Returns:
            DataFrame containing the transformed dataset.
        """
        logger.info("Starting Feature Engineering pipeline...")
                
        self.generate_transformations()
        
        # Apply transformations
        transformed_dataset = self.apply_transformations()
        
        if transformed_dataset is not None:
            logger.info("Pipeline completed successfully.")
        else:
            logger.error("Pipeline failed to produce transformed dataset.")

        return transformed_dataset, self.transformations
