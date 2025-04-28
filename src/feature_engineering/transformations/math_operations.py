import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.feature_engineering.transformations.base_transformation import BaseTransformation


class MathOperationsTransform(BaseTransformation):
    """
    Applies mathematical operations to numeric columns.
    """

    PROVIDER="math_operations"
    DESCRIPTION = """
    This transformation applies mathematical operations to numeric columns.

    Input:
        - source_columns: List of column names to process. Can be one or more columns.
        
    Output:
        - new_column_name: The name of the output column after applying the transformation.
        
    Param:
        - operation: The type of mathematical operation to apply. Supported operations are:
            - 'log': Logarithm (handles zeros and negative values by applying log1p).
            - 'sqrt': Square root (handles negative values by clipping to zero).
            - 'square': Square of the column values.
            - 'mean': Mean of the specified columns.
            - 'sum': Sum of the specified columns.
            - 'diff': Difference between two specified columns.
            - 'ratio': Ratio between two specified columns (handles division by zero).
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the math operations transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to process
            param: Dictionary containing the operation type ('log', 'sqrt', 'square', 'mean', 'sum', 'diff', 'ratio')
        """
        super().__init__(new_column_name, source_columns, param)
        
        # Validate param structure
        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with an 'operation' key.")
        
        if param["operation"] not in ["multiply", "add", "log", "sqrt", "square", "mean", "sum", "diff", "ratio"]:
            raise ValueError(f"Unsupported operation: {param['operation']}")
    
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
                if self.param["operation"] == 'multiply':
                    print("multiply")

                    # Multiply two columns
                    result_df[self.new_column_name] = df[self.source_columns].prod(axis=1)
                
                elif self.param["operation"] == 'add':
                    print("add")
                    # Calculate sum of columns
                    result_df[self.new_column_name] = df[self.source_columns].sum(axis=1)

            
            if len(valid_cols) == 2:
                col1, col2 = valid_cols
        
        return result_df
