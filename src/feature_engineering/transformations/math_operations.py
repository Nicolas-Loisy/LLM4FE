"""
Mathematical operations for feature engineering.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from src.feature_engineering.transformations.base_transformation import BaseTransformation


class MathOperationsTransform(BaseTransformation):
    """
    Applies mathematical operations to numeric columns.
    """
    def __init__(
        self,
        transformation_type: str,
        description: str,
        category: str,
        new_column_name: str,
        source_columns: List[str],
        transformation_params: Optional[str] = None
    ):
        """
        Initialize the math operations transformation.
        
        Args:
            - transformation_type: The name of the new transformation
            - description: The transformation description
            - category: The transformation category
            - new_column_name: The name of the new column
            - source_columns: List of columns to process
            - transformation_params: Optional Parameter (par défaut None)
        """
        super().__init__(transformation_type, description, category, new_column_name, source_columns, transformation_params)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mathematical operation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with new column added
        """
        result_df = df.copy()
        
        # Single column operations
        if len(self.source_columns) == 1 and self.source_columns[0] in df.columns:
            col = self.source_columns[0]
        
        # Multi-column operations
        elif len(self.source_columns) > 1:
            valid_cols = [col for col in self.source_columns if col in df.columns]
            
            if len(valid_cols) > 0:
                if self.transformation_type == 'multiply':
                    print("multiply")

                    # Multiply two columns
                    result_df[self.new_column_name] = df[self.source_columns].prod(axis=1)
                
                elif self.transformation_type == 'add':
                    print("add")
                    # Calculate sum of columns
                    result_df[self.new_column_name] = df[self.source_columns].sum(axis=1)

            
            if len(valid_cols) == 2:
                col1, col2 = valid_cols
        
        return result_df
