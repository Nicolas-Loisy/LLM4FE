"""
Encoding transformations for categorical variables.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import List, Optional
from .base_transform import BaseTransform


class EncodingTransform(BaseTransform):
    """
    Applies encoding transformations to categorical columns.
    """
    def __init__(self, final_col: str, cols_to_process: List[str], param: Optional[str] = 'onehot'):
        """
        Initialize the encoding transformation.
        
        Args:
            final_col: The name of the output column after transformation
            cols_to_process: List of column names to process
            param: Type of encoding to apply ('onehot', 'label', 'ordinal')
        """
        super().__init__(final_col, cols_to_process, param)
        self.encoder = None
        
        if param == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif param == 'label':
            self.encoder = LabelEncoder()
        # For ordinal encoding, we'll need to define the categories in the transform method
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with encoded column(s) added
        """
        result_df = df.copy()
        
        if self.param == 'onehot':
            # For one-hot encoding
            if len(self.cols_to_process) == 1:
                col = self.cols_to_process[0]
                if col in df.columns:
                    # Reshape for sklearn's fit_transform
                    values = df[col].values.reshape(-1, 1)
                    encoded = self.encoder.fit_transform(values)
                    
                    # Create new columns for each category
                    categories = self.encoder.categories_[0]
                    for i, category in enumerate(categories):
                        new_col = f"{self.final_col}_{category}"
                        result_df[new_col] = encoded[:, i]
            else:
                # Multiple columns for one-hot encoding
                valid_cols = [col for col in self.cols_to_process if col in df.columns]
                if valid_cols:
                    encoded = self.encoder.fit_transform(df[valid_cols])
                    
                    # Get all category names from all columns
                    all_categories = []
                    for i, categories in enumerate(self.encoder.categories_):
                        col_name = valid_cols[i]
                        for category in categories:
                            all_categories.append(f"{col_name}_{category}")
                    
                    # Create new columns for each category
                    for i, category in enumerate(all_categories):
                        new_col = f"{self.final_col}_{category}"
                        result_df[new_col] = encoded[:, i]
        
        elif self.param == 'label':
            # For label encoding (works on a single column)
            if len(self.cols_to_process) == 1:
                col = self.cols_to_process[0]
                if col in df.columns:
                    result_df[self.final_col] = self.encoder.fit_transform(df[col])
        
        elif self.param == 'ordinal':
            # For ordinal encoding, we need a mapping
            # This is a simplified version; in practice, you'd want to define the order
            if len(self.cols_to_process) == 1:
                col = self.cols_to_process[0]
                if col in df.columns:
                    # Get unique values and assign ordinal values
                    unique_values = df[col].unique()
                    mapping = {val: i for i, val in enumerate(unique_values)}
                    result_df[self.final_col] = df[col].map(mapping)
        
        return result_df
