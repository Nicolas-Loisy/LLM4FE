import pandas as pd
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class DateTimeProcessingTransform(BaseTransformation):
    """
    Applies date/time processing transformations to date columns.
    """

    PROVIDER = "datetime_processing"
    DESCRIPTION = """
    This transformation applies date/time-based processing to date columns.

    Input:
        - source_columns: List of column names to process (can be one or two columns).
        
    Output:
        - new_column_name: Name of the output column or prefix for multiple outputs.
        
    Param:
        - operation: The type of temporal operation to apply. Supported operations are:
            - 'year': Extract year from the date.
            - 'month': Extract month from the date.
            - 'day': Extract day from the date.
            - 'weekday': Extract the weekday from the date.
            - 'days_diff': Compute the difference in days between two date columns.
            - 'period': Categorize dates into predefined periods (param["periods"] expected).
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the date/time processing transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to process (can be one or two columns).
            param: Parameters for the transformation, e.g. 'operation' or 'periods'
        """
        super().__init__(new_column_name, source_columns, param)
        self.valid = True
        valid_operations = ['year', 'month', 'day', 'weekday', 'days_diff', 'period']

        if not isinstance(self.param, dict) or "operation" not in self.param:
            logger.error(f"[{self.PROVIDER}] Invalid param: expected a dictionary with an 'operation' key.")
            self.valid = False

        elif self.param["operation"] not in valid_operations:
            logger.error(f"[{self.PROVIDER}] Unsupported operation: {self.param['operation']}")
            self.valid = False

        elif len(source_columns) not in [1, 2]:
            logger.error(f"[{self.PROVIDER}] Invalid number of source columns: {len(source_columns)}. Must be 1 or 2.")
            self.valid = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply date/time processing transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed date column(s) added
        """
        result_df = df.copy()
        if not self.valid:
            logger.warning(f"[{self.PROVIDER}] Skipping transformation due to invalid configuration.")
            return result_df

        try :
            for col in self.source_columns:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found in dataframe. Skipping.")
                    return result_df
                if not pd.api.types.is_datetime64_any_dtype(result_df[col]):
                    logger.info(f"Converting '{col}' to datetime")
                    result_df[col] = pd.to_datetime(result_df[col], errors="coerce")


            col = self.source_columns[0]
            
            operation = self.param["operation"]
            
            if operation == 'year':
                logger.info("Extracting year")
                result_df[self.new_column_name] = result_df[col].dt.year

            elif operation == 'month':
                logger.info("Extracting month")
                result_df[self.new_column_name] = result_df[col].dt.month
            
            elif operation == 'day':
                logger.info("Extracting day")
                result_df[self.new_column_name] = result_df[col].dt.day

            elif operation == 'weekday':
                logger.info("Extracting weekday")
                result_df[self.new_column_name] = result_df[col].dt.weekday

            elif operation == 'days_diff':
                logger.info("Calculating date difference in days")
                if len(self.source_columns) < 2:
                    logger.warning("Need two columns for 'days_diff'. Skipping.")
                    return result_df
                second_col = self.source_columns[1]
                if second_col not in result_df.columns:
                    logger.warning(f"Second column '{second_col}' not found. Skipping.")
                    return result_df

                if not pd.api.types.is_datetime64_any_dtype(result_df[second_col]):
                    logger.info(f"Converting '{second_col}' to datetime")
                    result_df[second_col] = pd.to_datetime(result_df[second_col], errors="coerce")

                if result_df[second_col].isna().all():
                    logger.warning(f"All values in '{second_col}' could not be converted to datetime. Skipping.")
                    return result_df
                
                result_df[self.new_column_name] = (result_df[second_col] - result_df[col]).dt.days

            elif operation == 'period':
                logger.info("Grouping by standard time period")
                freq = self.param.get("freq")
                if not freq:
                    logger.warning("Missing 'freq' for 'period'. Skipping.")
                    return result_df
                logger.info("Grouping into periods")
                result_df[self.new_column_name] = result_df[col].dt.to_period(freq).astype(str)

            else:
                logger.warning(f"Unsupported operation: '{operation}'. Skipping.")

        except Exception as e:
            logger.exception(f"[{self.PROVIDER}] Unexpected error during transformation: {e}")
            return result_df
        
        return result_df