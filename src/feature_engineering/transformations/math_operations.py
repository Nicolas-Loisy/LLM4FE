import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class MathOperationsTransform(BaseTransformation):
    """
    Applies mathematical operations to numeric columns.
    """

    PROVIDER = "math_operations"
    DESCRIPTION = """
    This transformation applies mathematical operations to numeric columns.
    - columns_to_process: List of column names to process. Can be one or more columns.
    - new_column_name: The name of the output column after applying the transformation.
    - params:
        - operation: The type of mathematical operation to apply. Supported operations are:
            - 'log1p': Logarithm (handles zeros and negative values by applying log1p).
            - 'sqrt': Square root (handles negative values by clipping to zero).
            - 'square': Square of the column values.
            - 'mean': Mean of the specified columns.
            - 'sum': Sum of the specified columns.
            - 'difference': Difference between two specified columns.
            - 'ratio': Ratio between two specified columns (handles division by zero).
            - 'multiply': Product of the specified columns.
            **Important:** The operation names must match exactly as specified in the operation parameter.
    """

    def __init__(self, new_column_name: str, columns_to_process: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the math operations transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            columns_to_process: List of column names to process
            param: Dictionary containing the operation type ('log1p', 'sqrt', 'square', 'mean', 'sum', 'difference', 'ratio', 'multiply')
        """
        super().__init__(new_column_name, columns_to_process, param)
        
        # Validate param structure
        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with an 'operation' key.")
        
        if param["operation"] not in ["multiply", "sum", "log1p", "sqrt", "square", "mean", "difference", "ratio"]:
            raise ValueError(f"Unsupported operation: {param['operation']}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        try:
            # Single column operations
            if len(self.columns_to_process) == 1 :
                col = self.columns_to_process[0]
                if self.param["operation"] == 'log1p':
                    # Apply log1p to handle zeros and negative values
                    logger.debug(f"Applying log1p transformation to column '{col}'")
                    result_df[self.new_column_name] = np.log1p(df[col])
                
                elif self.param["operation"] == 'sqrt':
                    # Apply square root, clipping negative values to zero
                    logger.debug(f"Applying sqrt transformation to column '{col}'")
                    result_df[self.new_column_name] = np.sqrt(df[col].clip(lower=0))
                
                elif self.param["operation"] == 'square':
                    # Square the column values
                    logger.debug(f"Applying square transformation to column '{col}'")
                    result_df[self.new_column_name] = df[col] ** 2
            
            # Multi-column operations
            elif len(self.columns_to_process) > 1:
                valid_cols = [col for col in self.columns_to_process if col in df.columns]
                
                if len(valid_cols) > 0:
                    if self.param["operation"] == 'difference':
                        # Calculate difference between two columns
                        logger.debug(f"Calculating difference between columns {valid_cols[0]} and {valid_cols[1]}")
                        result_df[self.new_column_name] = df[valid_cols[0]] - df[valid_cols[1]]
                    
                    elif self.param["operation"] == 'ratio':
                        # Calculate ratio between two columns, handling division by zero
                        logger.debug(f"Calculating ratio between columns {valid_cols[0]} and {valid_cols[1]}")
                        result_df[self.new_column_name] = df[valid_cols[0]] / df[valid_cols[1]].replace(0, np.nan)
                    
                    elif self.param["operation"] == 'mean':
                        # Calculate mean of columns
                        logger.debug(f"Calculating mean across columns: {self.columns_to_process}")
                        result_df[self.new_column_name] = df[self.columns_to_process].mean(axis=1)

                    if self.param["operation"] == 'multiply':
                        # Multiply two columns
                        logger.debug(f"Multiplying columns: {self.columns_to_process}")        
                        result_df[self.new_column_name] = df[self.columns_to_process].prod(axis=1)
                    
                    elif self.param["operation"] == 'sum':
                        # Calculate sum of columns
                        logger.debug(f"Adding columns: {self.columns_to_process}")
                        result_df[self.new_column_name] = df[self.columns_to_process].sum(axis=1)
        except Exception as e:
            # If an unexpected error occurs, fill the entire column with NaN
            logger.error(f"Error during math operations transformation: {e}")
            result_df[self.new_column_name] = np.nan

        return result_df
