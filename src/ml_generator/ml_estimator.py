from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, root_mean_squared_error as rmse_score
)

import time
import pandas as pd
import logging

logger = logging.getLogger(__name__)

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
RMSE = 'RMSE'
F1SCORE = 'F1-Score' 

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

        if not self._load_dataset():
            raise ValueError(f"Failed to load dataset from {self.dataset_path}")

    def _load_dataset(self) -> bool:
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
            logger.info(f"Dataset loaded from {self.dataset_path}")
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

    def _determine_ml_problem_type(self):
        """Determine if this is a classification or regression problem."""
        if self.Y is None:
            logger.error("No target variable loaded.")
            return
        
        unique_values = self.Y.nunique()
        total_values = len(self.Y)
        unique_ratio = unique_values / total_values
        dtype = self.Y.dtype

        if dtype == 'bool':
            self.problem_type = CLASSIFICATION
        elif unique_values <= 10:
            self.problem_type = CLASSIFICATION
        elif dtype in ['int64', 'int32', 'int16']:
            if unique_values <= 20 and unique_ratio < 0.05:
                self.problem_type = CLASSIFICATION
            # If consecutive values begin with 0 or 1 (often encoded classes)
            elif unique_values < 30 and self.Y.min() in [0, 1] and \
                set(self.Y.unique()) == set(range(self.Y.min(), self.Y.max() + 1)):
                self.problem_type = CLASSIFICATION
            else:
                self.problem_type = REGRESSION
        else:
            self.problem_type = REGRESSION

        if self.problem_type == CLASSIFICATION:
            self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        else:
            self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        logger.info(f"Problem type detected: {self.problem_type}")
        logger.info(f"The model used depending on the ML problem: {self.model.__class__.__name__}")

    def _train_and_predict(self, test_size: float = 0.2) -> Dict[str, Any]:
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
        
        self._determine_ml_problem_type()
        
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.X, self.Y, 
                test_size=test_size, 
                random_state=42,
                stratify=self.Y if self.problem_type == CLASSIFICATION else None
            )

            train_time_start = time.time()
            self.model.fit(X_train, Y_train)
            ml_train_time = time.time() - train_time_start
            logger.info("Model training completed.")
            
            Y_prediction = self.model.predict(X_test)
            logger.info("Model prediction completed.")
            
            if self.problem_type == CLASSIFICATION:
                self.score = f1_score(Y_test, Y_prediction, average='macro')
                metric_name = F1SCORE
            else:
                self.score = rmse_score(Y_test, Y_prediction)
                metric_name = RMSE
            
            logger.info(f"{metric_name}: {self.score:.4f}")
            
            return {
                'problem_type': self.problem_type,
                'score': self.score,
                'metric_name': metric_name,
                'train_time': ml_train_time
            }
        except Exception as e:
            logger.error(f"Error during training and prediction: {e}")
            return {}
        
    @staticmethod
    def get_best_score(current_score: float, old_score: float, problem_type: str) -> Tuple[bool, float]:
        """
        Compare the current score with an old score to determine if the new dataset is better.
        
        Args:
            current_score: Current score to compare.
            old_score: Previous score to compare against.
            problem_type: Type of ML problem ('classification' or 'regression').
        
        Returns:
            Tuple containing (is_better: bool, normalized_percentage_change: float)
            - Normalized percentage: always positive when there's improvement, negative when worse
            - For classification (F1-score): higher score = positive percentage
            - For regression (RMSE): lower score = positive percentage (inverted logic)
            True if the new score is better, False otherwise.
        """
        if current_score is None:
            logger.warning("No current score available for comparison.")
            return False, 0.0
        
        if problem_type is None:
            logger.warning("Problem type not provided.")
            return False, 0.0
        
        if old_score == 0:
            logger.warning("Old score is zero, cannot compare.")
            return current_score > 0, float('inf') if current_score > 0 else 0.0
        
        percentage_change = ((current_score - old_score) / abs(old_score)) * 100

        if problem_type == CLASSIFICATION:
            # For F1-score, higher is better
            is_better = current_score >= old_score
            normalized_percentage = percentage_change
            comparison_symbol = ">=" if is_better else "<"
        else:
            # For RMSE, lower is better
            is_better = current_score <= old_score
            normalized_percentage = -percentage_change # Reverse the percentage to get a normal percentage (now a positive percentage is an improvement)
            comparison_symbol = "<=" if is_better else ">"
        
        logger.info(f"Score comparison: {current_score:.4f} {comparison_symbol} {old_score:.4f} "
                    f"Normalized improvement: {normalized_percentage:+.2f}% - "
                    f"({'Better/Same' if is_better else 'Worse'})")
        
        return is_better, normalized_percentage
        
    def run(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            test_size: Proportion of data to use for testing.
        
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting Machine Learning Estimator pipeline for a single iteration...")

        results = self._train_and_predict(test_size)
        
        if not results:
            logger.error("Machine learning Estimator failed to train and predict with the dataset.")
            return {}

        logger.info("Machine learning Estimator pipeline completed successfully !")
        
        return {
            **results
        }