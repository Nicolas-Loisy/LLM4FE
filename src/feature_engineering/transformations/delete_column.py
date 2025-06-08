import logging
import pandas as pd
from typing import List, Optional, Dict, Any

from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class DeleteColumnTransform(BaseTransformation):
    """
    Transformation that deletes specified columns from the dataframe.
    """

    PROVIDER = "delete_column"
    DESCRIPTION = """
    This transformation deletes one or more specified columns from the dataframe.
    - columns_to_process: The names of the columns to delete. Must be a list containing at least one column name.
    - params:
        - No additional parameters required.
    Output: The dataframe without the specified columns.
    """

    def __init__(self, new_column_name: str, columns_to_process: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the delete column transformation.
        
        Args:
            new_column_name: Not used here but kept for compatibility.
            columns_to_process: List containing column names to delete.
            param: No parameters needed.
        """
        super().__init__(new_column_name, columns_to_process, param)
        self.valid = True

        if not isinstance(columns_to_process, list):
            logger.error(f"[{self.PROVIDER}] 'columns_to_process' must be a list.")
            self.valid = False
        if len(columns_to_process) == 0:
            logger.error(f"[{self.PROVIDER}] At least one column must be provided for deletion.")
            self.valid = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the deletion of the columns from the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe without the specified columns.
        """
        result_df = df.copy()
        if not self.valid:
            logger.warning(f"[{self.PROVIDER}] Skipping transformation due to invalid configuration.")
            return result_df

        try : 
            cols_to_delete = self.columns_to_process
            
            # Check which columns exist in the dataframe
            existing_cols = [col for col in cols_to_delete if col in df.columns]
            missing_cols = [col for col in cols_to_delete if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"[{self.PROVIDER}] Columns {missing_cols} not found in dataframe and will be skipped.")
            
            if not existing_cols:
                logger.warning(f"[{self.PROVIDER}] No valid columns to delete.")
                return result_df
            
            result_df.drop(columns=existing_cols, inplace=True)
            logger.info(f"[{self.PROVIDER}] Successfully deleted columns: {existing_cols}")

            return result_df
        
        except Exception as e:
            logger.exception(f"[{self.PROVIDER}] Error during column deletion: {e}")
            return result_df