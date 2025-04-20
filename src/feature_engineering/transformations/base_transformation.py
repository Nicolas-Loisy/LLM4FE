"""
Base transformation class that all transformations will inherit from.
"""
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseTransform(ABC):
    """
    Abstract base class for all transformations.
    """
    def __init__(self, final_col: str, cols_to_process: List[str], param: Optional[str] = None):
        """
        Initialize the transformation.
        
        Args:
            final_col: The name of the output column after transformation
            cols_to_process: List of column names to process
            param: Optional parameter for the transformation
        """
        self.final_col = final_col
        self.cols_to_process = cols_to_process
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
        return f"{self.__class__.__name__}(final_col={self.final_col}, cols_to_process={self.cols_to_process}, param={self.param})"
