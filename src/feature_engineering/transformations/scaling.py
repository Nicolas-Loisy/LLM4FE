"""
Scaling transformations for feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Optional
from src.feature_engineering.transformations.base_transformation import BaseTransformation


class ScalingTransform(BaseTransformation):
    """
    Applies scaling transformations to numeric columns.
    """
    def __init__(self, final_col: str, cols_to_process: List[str], param: Optional[str] = 'standard'):
        """
        Initialize the scaling transformation.
        
        Args:
            final_col: The name of the output column after transformation
            cols_to_process: List of column names to process
            param: Type of scaling to apply ('standard', 'minmax', 'robust')
        """
        super().__init__(final_col, cols_to_process, param)
        self.scaler = None
        
        if param == 'standard':
            self.scaler = StandardScaler()
        elif param == 'minmax':
            self.scaler = MinMaxScaler()
        elif param == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()  # Default to standard scaling
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with scaled column added
        """
        if len(self.cols_to_process) == 1:
            # Single column transformation
            col = self.cols_to_process[0]
            if col in df.columns:
                # Reshape for sklearn's fit_transform
                values = df[col].values.reshape(-1, 1)
                df[self.final_col] = self.scaler.fit_transform(values).flatten()
        else:
            # Multiple columns transformation
            valid_cols = [col for col in self.cols_to_process if col in df.columns]
            if valid_cols:
                # Create a new column with the scaled values
                df[self.final_col] = self.scaler.fit_transform(df[valid_cols])[:, 0]
        
        return df
