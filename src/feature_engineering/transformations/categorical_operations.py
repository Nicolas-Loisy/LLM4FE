import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from sklearn.calibration import LabelEncoder    
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class CategoricalOperationsTransform(BaseTransformation):
    """
    Applies categorical operations to columns.
    """

    PROVIDER = "categorical_operations"
    DESCRIPTION = """
    This transformation applies categorical operations to columns of the dataset.
    - columns_to_process: List of column names to process. Can be one or more columns.
    - new_column_name: The name of the output column after applying the transformation.
    - params:
        - operation: The type of categorical operation to apply. Supported operations are:
            "encodage_oneHot", "label_encoding"
    """

    def __init__(self, new_column_name: str, columns_to_process: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the categorical operations transformation.
        Args:
            new_column_name: The name of the output column after transformation
            columns_to_process: List of column names to process
            param: Dictionary containing the operation type ('encodage_oneHot', 'label_encoding')
        """
        super().__init__(new_column_name, columns_to_process, param)

        # Validate param structure
        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with an 'operation' key.")

        if param["operation"] not in ["encodage_oneHot", "label_encoding"]:
            raise ValueError(f"Unsupported operation: {param['operation']}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the categorical operation to the DataFrame.
        """
        result_df = df.copy()

        if self.param["operation"] == 'encodage_oneHot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            for col in self.columns_to_process:
                try:
                    encoded_array = encoder.fit_transform(df[[col]])
                    new_columns = encoder.get_feature_names_out([col])
                    encoded_df = pd.DataFrame(encoded_array, columns=new_columns, index=df.index)
                    result_df = pd.concat([result_df, encoded_df], axis=1)
                except Exception as e:
                    logger.error(f"Error during one-hot encoding for column '{col}': {str(e)}")
                    # En cas d'erreur, remplir les colonnes avec des NaN
                    try:
                        for val in df[col].unique():
                            result_df[f"{col}_{val}"] = np.nan
                    except Exception as inner_e:
                        logger.error(f"Error creating fallback columns for '{col}': {str(inner_e)}")

        elif self.param["operation"] == 'label_encoding':
            for col in self.columns_to_process:
                encoder = LabelEncoder()
                try:
                    # Fit only on non-null values
                    not_null_mask = df[col].notnull()
                    result_df[self.new_column_name] = np.nan
                    result_df.loc[not_null_mask, self.new_column_name] = encoder.fit_transform(df.loc[not_null_mask, col])
                except Exception as e:
                    logger.error(f"Error during label encoding for column '{col}': {str(e)}")
                    result_df[self.new_column_name] = np.nan

        # elif self.param["operation"] == 'target_encoding':
        # TODO : To fix, target column should not be supposed and the description should explained target_encoding
        #     target_column = df.columns[-1] # On suppose que la derniere colonne est la colonne target
        #     for col in self.columns_to_process:
        #         try:
        #             # Calculate mean target values for each category
        #             category_means = df.groupby(col)[target_column].mean()
                    
        #             # Map the means to the original column values
        #             result_df[self.new_column_name] = df[col].map(category_means)
                    
        #             # Handle cases where original column has NaN values
        #             result_df.loc[df[col].isnull(), self.new_column_name] = np.nan
                    
        #         except Exception as e:
        #             logger.error(f"Error during target encoding for column '{col}': {str(e)}")
        #             result_df[self.new_column_name] = np.nan

        return result_df
