from pathlib import Path
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error
)

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MachineLearningEstimator:
    """
    Initialize Machine Learning Estimator focused on supervised machine learning with Random Forest for evaluating Feature Engineering impact.
    
    Args:
        dataset_path: Path to the dataset file (CSV format).
        target_col: Name of the target column in the dataset.
    """
    def __init__(self, dataset_path: str, target_col: str):
        self.dataset_path: Path = Path(dataset_path)
        self.target_col: str = target_col
        self.dataset: pd.DataFrame = None
        self.X: pd.DataFrame = None
        self.Y: pd.Series = None
        self.problem_type: str = None
        self.model = None
        self.score: float = None

        if not self.load_dataset():
            raise ValueError(f"Failed to load dataset from {self.dataset_path}")

    def load_dataset(self) -> bool:
        """
        Load the dataset from a CSV file.
        
        Args:
            dataset_path: Path to the CSV file.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"File not found: {self.dataset_path}")
            
            self.dataset = pd.read_csv(self.dataset_path)

            if self.target_col not in self.dataset.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in dataset")

            self.Y = self.dataset[self.target_col]
            self.X = self.dataset.drop(columns=[self.target_col])
            logger.info(f"Dataset successfully loaded: {self.dataset.shape}")
            logger.debug(f"Features shape: {self.X.shape}, Target shape: {self.Y.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def determine_ml_problem_type(self):
        """Determine if this is a classification or regression problem."""
        if self.Y is None:
            logger.error("No target variable loaded.")
            return
        
        unique_values = self.Y.nunique()
        total_values = len(self.Y)
        unique_ratio = unique_values / total_values
        dtype = self.Y.dtype

        if dtype == 'bool':
            self.problem_type = 'classification'
        elif unique_values <= 10:
            self.problem_type = 'classification'
        elif dtype in ['int64', 'int32', 'int16']:
            if unique_values <= 20 and unique_ratio < 0.05:
                self.problem_type = 'classification'
            # If consecutive values begin with 0 or 1 (often encoded classes)
            elif unique_values < 30 and self.Y.min() in [0, 1] and \
                set(self.Y.unique()) == set(range(self.Y.min(), self.Y.max() + 1)):
                self.problem_type = 'classification'
            else:
                self.problem_type = 'regression'
        else:
            self.problem_type = 'regression'

        if self.problem_type == 'classification':
            self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        else:
            self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        logger.info(f"Problem type detected: {self.problem_type}")
        logger.info(f"The model used depending on the ML problem: {self.model.__class__.__name__}")

    def train_and_predict(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train model and make predictions.
        
        Args:
            test_size: Proportion of data to use for testing.
            
        Returns:
            Dictionary containing results.
        """
        if self.X is None or self.Y is None:
            logger.error("No dataset loaded.")
            return {}
        
        self.determine_ml_problem_type()
        
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.X, self.Y, 
                test_size=test_size, 
                random_state=42,
                stratify=self.Y if self.problem_type == 'classification' else None
            )
            
            self.model.fit(X_train, Y_train)
            logger.info("Model training completed.")
            
            Y_prediction = self.model.predict(X_test)
            logger.info("Model prediction completed.")
            
            if self.problem_type == 'classification':
                self.score = accuracy_score(Y_test, Y_prediction)
                metric_name = "Accuracy"
            else:
                self.score = r2_score(Y_test, Y_prediction)
                metric_name = "R² Score"
            
            logger.info(f"{metric_name}: {self.score:.4f}")
            
            return {
                'problem_type': self.problem_type,
                'score': self.score,
                'metric_name': metric_name
            }
        except Exception as e:
            logger.error(f"Error during training and prediction: {e}")
            return {}

    def run(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            test_size: Proportion of data to use for testing.
        
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting Machine Learning Estimator pipeline for a single iteration...")

        results = self.train_and_predict(test_size)
        
        if not results:
            logger.error("Machine learning Estimator failed to train and predict with the dataset.")
            return {}

        logger.info("Machine learning Estimator pipeline completed successfully !")
        
        return {
            **results
        }