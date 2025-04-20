# Factory for managing transformations
from typing import Dict, Any, List, Optional
import pandas as pd

from src.feature_engineering.transformations.base_transformation import BaseTransformation
from src.feature_engineering.transformations.add_columns import AddColumnsTransform

class TransformationFactory:
    def __init__(self):
        self.transformations = {}
        self.provider_mapping = {
            'add': AddColumnsTransform,
            #'scaling': ScalingTransform,
            #'encoding': EncodingTransform,
            #'text': TextProcessingTransform,
            #'math': MathOperationsTransform,
            # Add more mappings as needed
        }

    def create_transformation(self, transformation_config: Dict[str, Any]) -> Optional[BaseTransformation]:
        """
        Create a transformation based on the configuration.
        
        Args:
            transformation_config: Dictionary containing transformation configuration
                - finalCol: The name of the output column
                - colToProcess: List of column names to process
                - providerTransform: Type of transformation ('math', 'encoding', 'scaling', etc.)
                - param: Optional parameter for the transformation
                
        Returns:
            A transformation instance or None if the provider is not found
        """
        final_col = transformation_config.get('finalCol')
        cols_to_process = transformation_config.get('colToProcess', [])
        provider = transformation_config.get('providerTransform')
        param = transformation_config.get('param')
        
        # Check if we have a valid provider
        if provider in self.provider_mapping:
            # Create the transformation
            transform_class = self.provider_mapping[provider]
            transformation = transform_class(final_col, cols_to_process, param)
            
            # Store the transformation
            transform_id = f"{provider}_{final_col}"
            self.transformations[transform_id] = transformation
            
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
