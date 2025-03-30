"""
Text processing transformations for feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Optional
from src.feature_engineering.transformations.base_transform import BaseTransform


class TextProcessingTransform(BaseTransform):
    """
    Applies text processing transformations to text columns.
    """
    def __init__(self, final_col: str, cols_to_process: List[str], param: Optional[str] = 'tfidf'):
        """
        Initialize the text processing transformation.
        
        Args:
            final_col: The name of the output column after transformation
            cols_to_process: List of column names to process
            param: Type of text processing to apply ('tfidf', 'count', 'length')
        """
        super().__init__(final_col, cols_to_process, param)
        self.vectorizer = None
        
        if param == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=100)
        elif param == 'count':
            self.vectorizer = CountVectorizer(max_features=100)
    
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
            if len(self.cols_to_process) == 1:
                col = self.cols_to_process[0]
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
                        new_col = f"{self.final_col}_{feature}"
                        result_df[new_col] = vectorized[:, i].toarray().flatten()
            
        elif self.param == 'length':
            # For text length
            if len(self.cols_to_process) == 1:
                col = self.cols_to_process[0]
                if col in df.columns:
                    # Calculate text length
                    result_df[self.final_col] = df[col].fillna('').astype(str).apply(len)
        
        return result_df
