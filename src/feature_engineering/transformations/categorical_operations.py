import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.calibration import LabelEncoder    
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering.transformations.base_transformation import BaseTransformation
class CategoricalOperationsTransform(BaseTransformation):

    """
    Applies categorical operations to columns.
    """

    PROVIDER="categorical_operations"
    DESCRIPTION = """
    This transformation applies categorical operations to columns of the dataset.

    Input:
        - source_columns: List of column names to process. Can be one or more columns.
        
    Output:
        - new_column_name: The name of the output column after applying the transformation.
        
    Param:
        - operation: The type of categorical operation to apply. Supported operations are:
        "Encodage One-Hot"
        "Label Encoding"
        "Target Encoding"
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the math operations transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to process
            param: Dictionary containing the operation type ('encodage_oneHot', 'label_encoding ', 'target_encoding')
        """
        super().__init__(new_column_name, source_columns, param)
        
        #Validate param structure
        if not isinstance(param, dict) or "operation" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with an 'operation' key.")
        
        if param["operation"] not in ["encodage_oneHot", "label_encoding", "target_encoding"]:
            raise ValueError(f"Unsupported operation: {param['operation']}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mathematical operation to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with new column added
        """
        result_df = df.copy()
    
        if self.param["operation"] == 'encodage_oneHot' :

            encoder = OneHotEncoder(sparse_output=False)
            for col in self.source_columns:
                endoded_array = encoder.fit_transform(df[[col]])
                new_columns = encoder.get_feature_names_out([col])
            # Créer un DataFrame pour les nouvelles colonnes encodées
                encoded_df = pd.DataFrame(endoded_array, columns=new_columns)
            # Concaténer les nouvelles colonnes avec le dataset existant
                result_df = pd.concat([result_df, encoded_df], axis=1)
          

        if self.param["operation"] == 'label_encoding' :
            encoder = LabelEncoder()
            for col in self.source_columns:
                result_df[self.new_column_name] = encoder.fit_transform(df[col])
        
        if self.param["operation"] == 'target_encoding' :
            # Assuming target column is the last column in the DataFrame
            target_column = df.columns[-1]
            for col in self.source_columns:
                # Calculate the mean of the target variable for each category
                mean_target = df.groupby(col)[target_column].mean()
                # Map the mean values to the original column
                result_df[self.new_column_name] = df[col].map(mean_target)

        return result_df
      

