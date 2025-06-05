"""
Text processing transformations for feature engineering.
"""
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class TextProcessingTransform(BaseTransformation):
    """
    Applies text processing transformations to text columns.
    """

    PROVIDER = "text_processing"
    DESCRIPTION = """
    This transformation applies text-based processing to text columns.

    Input:
        - source_columns: List of column names to process (only one column supported).
        
    Output:
        - new_column_name: Name of the output column or prefix for multiple outputs.
        
    Param:
        - operation: The type of textual operation to apply. Supported operations are:
            - 'length': Number of characters in the text.
            - 'word_count': Number of words in the text.
            - 'keyword': Detect presence of a keyword (param["keyword"])
            - 'tfidf': Apply TF-IDF encoding (param["max_features"] optionnel)
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the text processing transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to process
            param: Parameters for the encoding transformation, like 'onehot', 'label', 'ordinal'
        """
        super().__init__(new_column_name, source_columns, param)

        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param: expected a dictionary with an 'operation' key.")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text processing transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed text column(s) added
        """
        result_df = df.copy()
        
        try :
            if len(self.source_columns) != 1:
                raise ValueError("TextProcessingTransform supports only one source column.")

            col = self.source_columns[0]
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")

            texts = df[col].fillna('').astype(str)
            operation = self.param["operation"]

            if operation == 'length':
                logger.info("Applying text length operation")
                result_df[self.new_column_name] = texts.apply(len)

            elif operation == 'word_count':
                logger.info("Applying word count operation")
                result_df[self.new_column_name] = texts.apply(lambda x: len(x.split()))
            
            elif operation == 'keyword':
                logger.info("Applying keyword detection operation")
                keyword = self.param.get("keyword")
                if not keyword:
                    raise ValueError("Missing 'keyword' in param for keyword detection.")
                result_df[self.new_column_name] = texts.apply(lambda x: int(keyword.lower() in x.lower()))

            elif operation == 'tfidf':
                logger.info("Applying TF-IDF operation")

                max_features = self.param.get("max_features", 10)
                vectorizer = TfidfVectorizer(max_features=max_features)
                tfidf_matrix = vectorizer.fit_transform(texts)
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[
                    f"{self.new_column_name}_{feat}" for feat in vectorizer.get_feature_names_out()
                ])
                result_df = pd.concat([result_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

            else:
                raise ValueError(f"Unsupported operation: {self.param['operation']}")
            
        except Exception as e:
            logger.exception(f"[{self.PROVIDER}] Error during text processing transformation: {e}")
            raise
        
        return result_df
