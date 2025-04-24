import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseTransformation(ABC):
    """
    Abstract base class for all transformations.
    """
    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the transformation.
        
        Args:
            new_column_name: Name of the column to store the result of the transformation
            source_columns: List of columns to process
            param: Optional parameter for the transformation
        """
        self.new_column_name = new_column_name
        self.source_columns = source_columns
        self.param = param
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the transformation."""
        return f"{self.__class__.__name__}(new_column_name={self.new_column_name}, source_columns={self.source_columns}, param={self.param})"
