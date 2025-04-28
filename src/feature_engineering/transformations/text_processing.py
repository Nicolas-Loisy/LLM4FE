"""
Text processing transformations for feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Optional
from src.feature_engineering.transformations.base_transformation import BaseTransformation


class TextProcessingTransform(BaseTransformation):
    """
    Applies text processing transformations to text columns.
    """
    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[str] = 'tfidf'):
        """
        Initialize the text processing transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to process
            param: Parameters for the encoding transformation, like 'onehot', 'label', 'ordinal'
        """
        super().__init__(new_column_name, source_columns, param)
        self.vectorizer = None
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text processing transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed text column(s) added
        """
        result_df = df.copy()
        
        if self.param == 'tfidf' or self.param == 'count':
            # For vectorization (TF-IDF or Count)
            if len(self.source_columns) == 1:
                col = self.source_columns[0]
                if col in df.columns:
                    # Fill NA values with empty string
                    texts = df[col].fillna('').astype(str)
                    
                    # Transform the text
                    vectorized = self.vectorizer.fit_transform(texts)
                    
                    # Get feature names
                    try:
                        feature_names = self.vectorizer.get_feature_names_out()
                    except AttributeError:
                        # For older scikit-learn versions
                        feature_names = self.vectorizer.get_feature_names()
                    
                    # Create new columns for top features
                    for i, feature in enumerate(feature_names):
                        new_col = f"{self.new_column_name}_{feature}"
                        result_df[new_col] = vectorized[:, i].toarray().flatten()
            
        elif self.param == 'length':
            # For text length
            if len(self.source_columns) == 1:
                col = self.source_columns[0]
                if col in df.columns:
                    # Calculate text length
                    result_df[self.new_column_name] = df[col].fillna('').astype(str).apply(len)
        
        return result_df
