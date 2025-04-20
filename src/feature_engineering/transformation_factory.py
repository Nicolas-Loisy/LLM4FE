# Factory for managing transformations
from typing import Dict, Any, List, Optional
import pandas as pd

from src.feature_engineering.transformations.base_transformation import BaseTransformation
from src.feature_engineering.transformations.math_operations import MathOperationsTransform

class TransformationFactory:
    def __init__(self):
        self.transformations = {}
        self.provider_mapping = {
            #'scaling': ScalingTransform,
            #'encoding': EncodingTransform,
            #'text': TextProcessingTransform,
            'math': MathOperationsTransform,
            # Add more mappings as needed
        }

    def create_transformation(self, transformation_config: Dict[str, Any]) -> Optional[BaseTransformation]:
        """
        Create a transformation based on the configuration.
        
        Args:
            transformation_config: Dictionnaire contenant la configuration de la transformation
                - transformation_type: The name of the new transformation
                - description: The transformation description
                - category: The transformation category
                - new_column_name: The name of the new column
                - source_columns: List of columns to process
                - transformation_params: Optional Parameter (par défaut None)
                
        Returns:
            A transformation instance or None if the provider is not found
        """

        transformation_type = transformation_config.get('transformation_type')
        description = transformation_config.get('description')
        category = transformation_config.get('category')
        new_column_name = transformation_config.get('new_column_name')
        source_columns = transformation_config.get('source_columns', [])
        transformation_params = transformation_config.get('transformation_params')
      
        if category in self.provider_mapping:
            transform_class = self.provider_mapping[category]
            transformation = transform_class(
                transformation_type,
                description,
                category,
                new_column_name,
                source_columns,
                transformation_params
            )

            transform_id = f"{transformation_type}_{new_column_name}"
            self.transformations[transform_id] = transformation

            print("transfo", transformation)
            return transformation
        
        return None
    
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered transformations to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        result_df = df.copy()
        
        for _, transformation in self.transformations.items():
            result_df = transformation.transform(result_df)
            
        return result_df
