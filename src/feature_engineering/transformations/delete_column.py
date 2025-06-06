"""
Delete a specified column from a dataframe.
"""
import logging
import pandas as pd
from typing import List, Optional, Dict, Any

from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class DeleteColumnTransform(BaseTransformation):
    """
    Transformation that deletes a specified column from the dataframe.
    """

    PROVIDER = "delete_column"
    DESCRIPTION = """
    This transformation deletes one specified column from the dataframe.

    Input:
        - source_columns: The name of the column to delete.
        
    Output:
        - The dataframe without the specified column.
        
    Param:
        - No additional parameters required.
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the delete column transformation.
        
        Args:
            new_column_name: Not used here but kept for compatibility.
            source_columns: List containing exactly one column name to delete.
            param: No parameters needed.
        """
        super().__init__(new_column_name, source_columns, param)
        self.valid = True

        if not isinstance(source_columns, list):
            logger.error(f"[{self.PROVIDER}] 'source_columns' must be a list.")
            self.valid = False
        if len(source_columns) != 1:
            logger.error(f"[{self.PROVIDER}] Exactly one source column must be provided, got {len(source_columns)}.")
            self.valid = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the deletion of the column from the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe without the specified column.
        """
        result_df = df.copy()
        if not self.valid:
            logger.warning(f"[{self.PROVIDER}] Skipping transformation due to invalid configuration.")
            return result_df

        try : 
            col_to_delete = self.source_columns[0]

            if col_to_delete not in df.columns:
                logger.error(f"[{self.PROVIDER}] Column '{col_to_delete}' not found in dataframe.")
                return result_df
            
            result_df.drop(columns=[col_to_delete], inplace=True)

            return result_df
        
        except Exception as e:
            logger.exception(f"[{self.PROVIDER}] Error during column deletion: {e}")
            return result_df