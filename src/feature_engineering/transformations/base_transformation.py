import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseTransformation(ABC):
    """
    Abstract base class for all transformations.
    """
    def __init__(self, 
                 transformation_type: str,
                 description: str,
                 category: str,
                 new_column_name: str,
                 source_columns: List[str],
                 transformation_params: Optional[dict] = None):
        self.transformation_type = transformation_type                          # Nom de la transformation
        self.description = description            # Description de ce qu'elle fait
        self.category = category                  # Catégorie de la transformation (ex: math, text, etc.)
        self.new_column_name = new_column_name        # Nom de la colonne de sortie
        self.source_columns = source_columns    # Colonnes concernées (source)
        self.transformation_params = transformation_params or {}                  # Paramètres supplémentaires (optionnels)
    
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
        return f"{self.__class__.__name__}(new_column_name={self.new_column_name}, source_columns={self.source_columns}, param={self.transformation_params})"
