# src/feature_engineering/transformations/add_columns.py
import pandas as pd
from src.feature_engineering.transformations.base_transformation import BaseTransformation

class AddColumnsTransform(BaseTransformation):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les colonnes spécifiées et calcule la nouvelle colonne.
        """
        df[self.final_col] = df[self.cols_to_process].sum(axis=1)
        return df
