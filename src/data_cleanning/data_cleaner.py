from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class DataCleaner:
    def clean_dataset(self, df: pd.DataFrame, threshold: float = 0.8):
        df_cleaned = df.copy()

        # Supprimer colonnes avec +80% de NaN
        missing_ratio = df_cleaned.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        df_cleaned.drop(columns=cols_to_drop, inplace=True)
        print(f"Colonnes supprimées (>{threshold*100}% NaN): {list(cols_to_drop)}")

        # Colonnes numériques / catégorielles
        num_cols = df_cleaned.select_dtypes(include=['number']).columns
        cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns

        # Remplacement des NaN
        for col in num_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

        for col in cat_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

        # Encodage
        for col in cat_cols:
            unique_vals = df_cleaned[col].nunique()
            if unique_vals <= 5:
                # One-hot encoding
                dummies = pd.get_dummies(df_cleaned[col], prefix=col)
                df_cleaned.drop(columns=col, inplace=True)
                df_cleaned = pd.concat([df_cleaned, dummies], axis=1)
            else:
                # Label encoding
                le = LabelEncoder()
                df_cleaned[col] = le.fit_transform(df_cleaned[col])

        return df_cleaned



# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de DataFrame
    data = pd.read_csv(Path("data/datasets/Tuberculosis_Dataset.csv"))
    cleaner = DataCleaner()
    df_clean = cleaner.clean_dataset(data)
    print(df_clean)