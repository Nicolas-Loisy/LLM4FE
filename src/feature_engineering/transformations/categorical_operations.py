import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.calibration import LabelEncoder    
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering.transformations.base_transformation import BaseTransformation
class CategoricalOperationsTransform(BaseTransformation):
    """
    Applies categorical operations to columns.
    """

    PROVIDER = "categorical_operations"
    DESCRIPTION = """
    This transformation applies categorical operations to columns of the dataset.

    Input:
        - source_columns: List of column names to process. Can be one or more columns.
        
    Output:
        - new_column_name: The name of the output column after applying the transformation.
        
    Param:
        - operation: The type of categorical operation to apply. Supported operations are:
        "encodage_oneHot", "label_encoding", "target_encoding"
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        super().__init__(new_column_name, source_columns, param)

        # Validate param structure
        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with an 'operation' key.")

        if param["operation"] not in ["encodage_oneHot", "label_encoding", "target_encoding"]:
            raise ValueError(f"Unsupported operation: {param['operation']}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()

        if self.param["operation"] == 'encodage_oneHot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            for col in self.source_columns:
                try:
                    encoded_array = encoder.fit_transform(df[[col]])
                    new_columns = encoder.get_feature_names_out([col])
                    encoded_df = pd.DataFrame(encoded_array, columns=new_columns, index=df.index)
                    result_df = pd.concat([result_df, encoded_df], axis=1)
                except Exception as e:
                    # En cas d'erreur, remplir les colonnes avec des NaN
                    for val in df[col].unique():
                        result_df[f"{col}_{val}"] = np.nan

        elif self.param["operation"] == 'label_encoding':
            for col in self.source_columns:
                encoder = LabelEncoder()
                try:
                    # Fit only on non-null values
                    not_null_mask = df[col].notnull()
                    result_df[self.new_column_name] = np.nan
                    result_df.loc[not_null_mask, self.new_column_name] = encoder.fit_transform(df.loc[not_null_mask, col])
                except Exception:
                    result_df[self.new_column_name] = np.nan

        elif self.param["operation"] == 'target_encoding':
            target_column = df.columns[-1] # On suppose que la derniere colonne est la colonne target
            for col in self.source_columns:
                result_df[self.new_column_name] = df.apply(
                    lambda row: df.groupby(col)[target_column].mean().get(row[col], np.nan)
                    if pd.notnull(row[col]) and pd.notnull(row[target_column])
                    else np.nan,
                    axis=1
                )

        return result_df

      

