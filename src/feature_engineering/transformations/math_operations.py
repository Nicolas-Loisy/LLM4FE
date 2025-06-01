import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.feature_engineering.transformations.base_transformation import BaseTransformation

class MathOperationsTransform(BaseTransformation):
    """
    Applies mathematical operations to numeric columns.
    """

    PROVIDER = "math_operations"
    DESCRIPTION = """
    This transformation applies mathematical operations to numeric columns.

    Input:
        - source_columns: List of column names to process. Can be one or more columns.
        
    Output:
        - new_column_name: The name of the output column after applying the transformation.
        
    Param:
        - operation: The type of mathematical operation to apply. Supported operations are:
            - 'log': Logarithm (handles zeros and negative values by applying log1p).
            - 'sqrt': Square root (handles negative values by clipping to zero).
            - 'square': Square of the column values.
            - 'mean': Mean of the specified columns.
            - 'sum': Sum of the specified columns.
            - 'diff': Difference between two specified columns.
            - 'ratio': Ratio between two specified columns (handles division by zero).
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        super().__init__(new_column_name, source_columns, param)

        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with an 'operation' key.")
        
        if param["operation"] not in ["multiply", "addition", "log", "sqrt", "square", "mean", "sum", "diff", "ratio"]:
            raise ValueError(f"Unsupported operation: {param['operation']}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()

        try : 

            if len(self.source_columns) == 1:
                col = self.source_columns[0]

                if self.param["operation"] == 'log':
                    result_df[self.new_column_name] = df[col].apply(
                        lambda x: np.log1p(x) if pd.notnull(x) and x > -1 else np.nan
                    )

                elif self.param["operation"] == 'sqrt':
                    result_df[self.new_column_name] = df[col].apply(
                        lambda x: np.sqrt(x) if pd.notnull(x) and x >= 0 else np.nan
                    )

                elif self.param["operation"] == 'square':
                    result_df[self.new_column_name] = df[col].apply(
                        lambda x: x ** 2 if pd.notnull(x) else np.nan
                    )

            elif len(self.source_columns) > 1:
                cols = self.source_columns
                if self.param["operation"] == 'diff' and len(cols) >= 2:
                    result_df[self.new_column_name] = df.apply(
                        lambda row: row[cols[0]] - row[cols[1]]
                        if pd.notnull(row[cols[0]]) and pd.notnull(row[cols[1]])
                        else np.nan, axis=1
                    )

                elif self.param["operation"] == 'ratio' and len(cols) >= 2:
                    result_df[self.new_column_name] = df.apply(
                        lambda row: row[cols[0]] / row[cols[1]]
                        if pd.notnull(row[cols[0]]) and pd.notnull(row[cols[1]]) and row[cols[1]] != 0
                        else np.nan, axis=1
                    )

                elif self.param["operation"] == 'mean':
                    result_df[self.new_column_name] = df[cols].apply(
                        lambda row: row.mean() if row.notnull().all() else np.nan, axis=1
                    )

                elif self.param["operation"] in ['sum', 'addition']:
                    result_df[self.new_column_name] = df[cols].apply(
                        lambda row: row.sum() if row.notnull().all() else np.nan, axis=1
                    )

                elif self.param["operation"] == 'multiply':
                    result_df[self.new_column_name] = df[cols].apply(
                        lambda row: row.prod() if row.notnull().all() else np.nan, axis=1
                    )

        except Exception:
            # Si une erreur inattendue survient, on remplit toute la colonne avec NaN
            result_df[self.new_column_name] = np.nan


        return result_df
