"""
Mathematical operations for feature engineering.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from src.feature_engineering.transformations.base_transform import BaseTransform


class MathOperationsTransform(BaseTransform):
    """
    Applies mathematical operations to numeric columns.
    """
    def __init__(self, final_col: str, cols_to_process: List[str], param: Optional[str] = 'log'):
        """
        Initialize the math operations transformation.
        
        Args:
            final_col: The name of the output column after transformation
            cols_to_process: List of column names to process
            param: Type of math operation to apply ('log', 'sqrt', 'square', 'mean', 'sum', 'diff', 'ratio')
        """
        super().__init__(final_col, cols_to_process, param)
    
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
        if len(self.cols_to_process) == 1 and self.cols_to_process[0] in df.columns:
            col = self.cols_to_process[0]
            
            if self.param == 'log':
                # Apply log transformation (handle zeros and negative values)
                result_df[self.final_col] = np.log1p(df[col].clip(lower=0))
            
            elif self.param == 'sqrt':
                # Apply square root (handle negative values)
                result_df[self.final_col] = np.sqrt(df[col].clip(lower=0))
            
            elif self.param == 'square':
                # Apply square
                result_df[self.final_col] = df[col] ** 2
        
        # Multi-column operations
        elif len(self.cols_to_process) > 1:
            valid_cols = [col for col in self.cols_to_process if col in df.columns]
            
            if len(valid_cols) > 0:
                if self.param == 'mean':
                    # Calculate mean of columns
                    result_df[self.final_col] = df[valid_cols].mean(axis=1)
                
                elif self.param == 'sum':
                    # Calculate sum of columns
                    result_df[self.final_col] = df[valid_cols].sum(axis=1)
            
            if len(valid_cols) == 2:
                col1, col2 = valid_cols
                
                if self.param == 'diff':
                    # Calculate difference between two columns
                    result_df[self.final_col] = df[col1] - df[col2]
                
                elif self.param == 'ratio':
                    # Calculate ratio between two columns (handle division by zero)
                    denominator = df[col2].replace(0, np.nan)
                    result_df[self.final_col] = df[col1] / denominator
                    # Fill NaN values with 0 or another appropriate value
                    result_df[self.final_col] = result_df[self.final_col].fillna(0)
        
        return result_df
