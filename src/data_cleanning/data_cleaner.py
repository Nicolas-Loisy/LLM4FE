from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def clean_dataset(self, df: pd.DataFrame, threshold: float = 0.8, target_column: str = None):
        df_cleaned = df.copy()

        # Supprimer colonnes avec +80% de NaN
        missing_ratio = df_cleaned.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        df_cleaned.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Colonnes supprimées (>{threshold*100}% NaN): {list(cols_to_drop)}")

        # Transformer les colonnes booléennes en binaire (0/1)
        bool_cols = df_cleaned.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_cleaned[col] = df_cleaned[col].astype(int)
            logger.info(f"Colonne booléenne '{col}' transformée en binaire")

        # Colonnes numériques / catégorielles
        num_cols = df_cleaned.select_dtypes(include=['number']).columns
        cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns

        # Nettoyage des valeurs infinies et aberrantes pour les colonnes numériques
        for col in num_cols:
            if df_cleaned[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Détecter les valeurs infinies
                inf_count = np.isinf(df_cleaned[col]).sum()
                if inf_count > 0:
                    logger.warning(f"Colonne '{col}': {inf_count} valeurs infinies détectées")
                    # Remplacer les valeurs infinies par NaN
                    df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)
                
                # Détecter les valeurs trop grandes (> float32 max)
                float32_max = np.finfo(np.float32).max
                large_values = (df_cleaned[col].abs() > float32_max).sum()
                if large_values > 0:
                    logger.warning(f"Colonne '{col}': {large_values} valeurs trop grandes pour float32")
                    # Clipper les valeurs trop grandes
                    df_cleaned[col] = df_cleaned[col].clip(-float32_max, float32_max)

        # Remplacement des NaN
        for col in num_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        for col in cat_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

        # Encoder la colonne target si spécifiée
        if target_column and target_column in df_cleaned.columns:
            if df_cleaned[target_column].dtype == 'object' or df_cleaned[target_column].dtype.name == 'category':
                target_encoder = LabelEncoder()
                df_cleaned[target_column] = target_encoder.fit_transform(df_cleaned[target_column])
                logger.info(f"Target column '{target_column}' encodée avec LabelEncoder")

        # Encodage des autres colonnes catégorielles (exclure la target)
        cat_cols_to_encode = [col for col in cat_cols if col != target_column]
        for col in cat_cols_to_encode:
            unique_vals = df_cleaned[col].nunique()
            if unique_vals <= 5:
                # One-hot encoding
                # - Génère des variables indicatrices pour 'col' avec préfixe (`prefix=col`), 
                # - conserve toutes les catégories (`drop_first=False`), 
                # - encode en entier : 0 / 1 (`dtype=int`)
                dummies = pd.get_dummies(df_cleaned[col], prefix=col, dummy_na=False, drop_first=False, dtype=int)
                df_cleaned.drop(columns=col, inplace=True)
                df_cleaned = pd.concat([df_cleaned, dummies], axis=1)
            else:
                # Label encoding
                le = LabelEncoder()
                df_cleaned[col] = le.fit_transform(df_cleaned[col])

        return df_cleaned

    def run(self, input_path: str, output_path: str = None, threshold: float = 0.8, target_column: str = None):
        """
        Execute the complete data cleaning pipeline
        
        Args:
            input_path: Path to the input CSV file
            output_path: Path to save the cleaned dataset (optional)
            threshold: Threshold for dropping columns with missing values
            target_column: Name of the target column for special handling
            
        Returns:
            str: Path to the cleaned dataset file
        """
        logger.info(f"Loading dataset from: {input_path}")
        data = pd.read_csv(Path(input_path))
        logger.info(f"Original dataset shape: {data.shape}")
        
        logger.info("Cleaning dataset...")
        df_clean = self.clean_dataset(data, threshold, target_column)
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        
        if not output_path:
            input_stem = Path(input_path).stem
            output_path = f"{input_stem}_cleaned.csv"
        
        df_clean.to_csv(Path(output_path), index=False)
        logger.info(f"Cleaned dataset saved to: {output_path}")
        
        return output_path
