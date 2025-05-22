"""
Encoding transformations for categorical variables.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import List, Optional
from src.feature_engineering.transformations.base_transformation import BaseTransformation


class EncodingTransform(BaseTransformation):
    """
    Applies encoding transformations to categorical columns.
    """

    PROVIDER = "encoding"
    DESCRIPTION = """
    This transformation applies encoding methods to categorical columns.

    Input:
        - source_columns: List of categorical column names to encode (one or more).
        
    Output:
        - new_column_name: The base name or prefix for the encoded output columns.
        
    Param:
        - operation: The encoding type to apply. Supported operations are:
            - 'onehot': One-hot encoding (creates multiple binary columns).
            - 'label': Label encoding (assigns integer labels).
            - 'ordinal': Ordinal encoding (maps categories to ordered integers).
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[str] = 'onehot'):
        """
        Initialize the encoding transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to process
            param: Parameters for the encoding transformation, like 'onehot', 'label', 'ordinal'
        """
        super().__init__(new_column_name, source_columns, param)
        self.encoder = None
    
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
            if len(self.source_columns) == 1:
                col = self.source_columns[0]
                if col in df.columns:
                    # Reshape for sklearn's fit_transform
                    values = df[col].values.reshape(-1, 1)
                    encoded = self.encoder.fit_transform(values)
                    
                    # Create new columns for each category
                    categories = self.encoder.categories_[0]
                    for i, category in enumerate(categories):
                        new_col = f"{self.new_column_name}_{category}"
                        result_df[new_col] = encoded[:, i]
            else:
                # Multiple columns for one-hot encoding
                valid_cols = [col for col in self.source_columns if col in df.columns]
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
                        new_col = f"{self.new_column_name}_{category}"
                        result_df[new_col] = encoded[:, i]
        
        elif self.param == 'label':
            # For label encoding (works on a single column)
            if len(self.source_columns) == 1:
                col = self.source_columns[0]
                if col in df.columns:
                    result_df[self.new_column_name] = self.encoder.fit_transform(df[col])
        
        elif self.param == 'ordinal':
            # For ordinal encoding, we need a mapping
            # This is a simplified version; in practice, you'd want to define the order
            if len(self.source_columns) == 1:
                col = self.source_columns[0]
                if col in df.columns:
                    # Get unique values and assign ordinal values
                    unique_values = df[col].unique()
                    mapping = {val: i for i, val in enumerate(unique_values)}
                    result_df[self.new_column_name] = df[col].map(mapping)
        
        return result_df
