import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.feature_engineering.transformations.base_transformation import BaseTransformation

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
        - 'convert': Convert the source columns to datetime format.
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

        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param: expected a dictionary with an 'operation' key.")

        if len(source_columns) not in [1, 2]:
            raise ValueError("DateTimeProcessingTransform supports either one or two source columns.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply date/time processing transformation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed date column(s) added
        """
        result_df = df.copy()

        for col in self.source_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if not pd.api.types.is_datetime64_any_dtype(result_df[col]):
                print(f"Converting '{col}' to datetime")
                result_df[col] = pd.to_datetime(result_df[col], errors="coerce")


        col = self.source_columns[0]
                
        if self.param["operation"] == 'year':
            print("Extracting year")
            result_df[self.new_column_name] = result_df[col].dt.year

        elif self.param["operation"] == 'month':
            print("Extracting month")
            result_df[self.new_column_name] = result_df[col].dt.month
        
        elif self.param["operation"] == 'day':
            print("Extracting day")
            result_df[self.new_column_name] = result_df[col].dt.day

        elif self.param["operation"] == 'weekday':
            print("Extracting weekday")
            result_df[self.new_column_name] = result_df[col].dt.weekday

        elif self.param["operation"] == 'days_diff':
            print("Calculating date difference in days")
            if len(self.source_columns) < 2:
                raise ValueError("Two date columns are required for 'days_diff' operation.")
            second_col = self.source_columns[1]
            result_df[self.new_column_name] = (result_df[second_col] - result_df[col]).dt.days

        elif self.param["operation"] == 'period':
            print("Grouping by standard time period")
            freq = self.param.get("freq")
            if not freq:
                raise ValueError("Missing 'freq' in param for period grouping.")
            result_df[self.new_column_name] = result_df[col].dt.to_period(freq).astype(str)

        else:
            raise ValueError(f"Unsupported operation: {self.param['operation']}")

        return result_df
