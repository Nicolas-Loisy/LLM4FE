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
    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[str] = 'standard'):
        """
        Initialize the scaling transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to process
            param: Parameters for the encoding transformation, like 'onehot', 'label', 'ordinal'
        """
        super().__init__(new_column_name, source_columns, param)
        self.scaler = None
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with scaled column added
        """
        if len(self.source_columns) == 1:
            # Single column transformation
            col = self.source_columns[0]
            if col in df.columns:
                # Reshape for sklearn's fit_transform
                values = df[col].values.reshape(-1, 1)
                df[self.new_column_name] = self.scaler.fit_transform(values).flatten()
        else:
            # Multiple columns transformation
            valid_cols = [col for col in self.source_columns if col in df.columns]
            if valid_cols:
                # Create a new column with the scaled values
                df[self.new_column_name] = self.scaler.fit_transform(df[valid_cols])[:, 0]
        
        return df
