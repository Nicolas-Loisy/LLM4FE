import logging
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline
from src.data_cleanning.data_cleaner import DataCleaner
from src.orchestrator.version_manager import VersionManager
from src.ml_generator.ml_estimator import MachineLearningEstimator
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config_path: str = "data/config.json"):
        """Initialize the orchestrator."""
        logger.info("Initializing Orchestrator...")
        self.config = get_config(config_path)
        self.version_manager = VersionManager() # Définir le dossier de save des versions
        self.data_cleaner = DataCleaner()
        self.best_score = -float('inf')
        self.best_dataset_path = None
        self.best_version = None

    def run(
        self, 
        dataset_path: str, 
        dataset_description: Optional[str] = None, 
        target_column: Optional[str] = None,
        iterations: int = 1
    ) -> Dict[str, Any]:
        """Main entry point to run the feature engineering pipeline."""
        logger.info(f"Starting LLM4FE orchestration with {iterations} iterations...")
        
        if target_column is None:
            raise ValueError("target_column is required for ML evaluation")
        
        try:
            # Clean the original dataset first
            logger.info("Cleaning original dataset...")
            cleaned_dataset_path = self._clean_original_dataset(dataset_path, target_column)
            
            current_dataset_path = cleaned_dataset_path
            current_description = dataset_description
            all_transformations = []
            
            # Initialize with cleaned dataset score
            logger.info("Evaluating baseline cleaned dataset...")
            baseline_score = self._evaluate_dataset(current_dataset_path, target_column)
            self.best_score = baseline_score
            self.best_dataset_path = current_dataset_path
            self.best_version = 0
            
            logger.info(f"Baseline score (cleaned): {baseline_score:.4f}")
            
            for i in range(iterations):
                logger.info(f"Starting iteration {i+1}/{iterations}...")
                logger.info(f"Using dataset: {current_dataset_path}")
                
                result = self._run_single_iteration(
                    current_dataset_path, 
                    current_description, 
                    target_column, 
                    all_transformations,
                    i + 1
                )
                
                if result is None:
                    logger.error(f"Iteration {i+1} failed")
                    break
                
                iteration_dataset_path, current_description, all_transformations, ml_score = result
                
                # Always use the last generated dataset for next iteration
                current_dataset_path = iteration_dataset_path
                
                # Update best score tracking
                if ml_score > self.best_score:
                    logger.info(f"New best score found: {ml_score:.4f} (previous: {self.best_score:.4f})")
                    self.best_score = ml_score
                    self.best_dataset_path = iteration_dataset_path
                    self.best_version = i + 1
                else:
                    logger.info(f"Score {ml_score:.4f} did not improve best score {self.best_score:.4f}")
                    logger.info(f"Continuing with last generated dataset for next iteration")
                
                logger.info(f"Iteration {i+1} completed successfully")
            
            self.version_manager.save_global_summary()
            logger.info(f"Completed all {iterations} iterations.")
            logger.info(f"Best score: {self.best_score:.4f} from version {self.best_version}")
            logger.info(f"Final dataset: {current_dataset_path}")
            
            return self._build_final_result(all_transformations, current_dataset_path)
            
        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            raise

    def _clean_original_dataset(self, dataset_path: str, target_column: str) -> str:
        """Clean the original dataset and return the path to cleaned dataset."""
        logger.info("Cleaning original dataset before feature engineering...")
        
        # Create a cleaned version of the original dataset
        cleaned_output_path = self.version_manager.save_dataset(
            pd.read_csv(dataset_path), "_baseline_cleaned"
        )
        
        # Apply data cleaning to original dataset
        final_cleaned_path = self.data_cleaner.run(
            input_path=dataset_path,
            output_path=cleaned_output_path,
            threshold=0.8,
            target_column=target_column
        )
        
        logger.info(f"Original dataset cleaned and saved to: {final_cleaned_path}")
        return final_cleaned_path

    def _run_single_iteration(
        self,
        dataset_path: str,
        description: Optional[str],
        target_column: str,
        all_transformations: List,
        iteration_number: int
    ) -> Optional[Tuple[str, str, List, float]]:
        """Run a single iteration of feature engineering and cleaning."""
        version = self.version_manager.increment_version()
        logger.info(f"Processing version {version}...")
        
        # Feature Engineering
        transformed_dataset, new_transformations, updated_description = self._run_feature_engineering(
            dataset_path, description, target_column
        )
        
        if transformed_dataset is None:
            return None
        
        # Save dataset after feature engineering
        fe_output_path = self.version_manager.save_dataset(transformed_dataset, "_fe")
        
        # Data Cleaning
        final_output_path = self._run_data_cleaning(fe_output_path, target_column)
        final_dataset = pd.read_csv(final_output_path)
        
        # ML Evaluation
        ml_score = self._evaluate_dataset(final_output_path, target_column)
        logger.info(f"ML Score for iteration {iteration_number}: {ml_score:.4f}")
        
        # Update tracking
        updated_description = updated_description or description or ""
        all_transformations.extend(new_transformations)
        
        # Save version information with ML score
        self._save_version_info(
            dataset_path, fe_output_path, final_output_path,
            new_transformations, all_transformations, updated_description,
            final_dataset, target_column, ml_score
        )
        
        return final_output_path, updated_description, all_transformations, ml_score

    def _evaluate_dataset(self, dataset_path: str, target_column: str) -> float:
        """Evaluate dataset using ML estimator and return score."""
        try:
            ml_estimator = MachineLearningEstimator(
                dataset_path=dataset_path,
                target_col=target_column
            )
            results = ml_estimator.run()
            return results.get('score', 0.0)
        except Exception as e:
            logger.error(f"ML evaluation failed: {str(e)}")
            return 0.0

    def _run_feature_engineering(
        self, 
        dataset_path: str, 
        description: Optional[str], 
        target_column: Optional[str]
    ) -> Tuple[Optional[pd.DataFrame], List, Optional[str]]:
        """Execute feature engineering pipeline."""
        fe_pipeline = FeatureEngineeringPipeline(
            dataset_path=dataset_path,
            dataset_description=description,
            target_column=target_column
        )
        return fe_pipeline.run()

    def _run_data_cleaning(self, input_path: str, target_column: Optional[str]) -> str:
        """Execute data cleaning pipeline."""
        logger.info("Applying data cleaning after feature engineering...")
        cleaned_output_path = self.version_manager.save_dataset(
            pd.read_csv(input_path), ""
        ).replace(".csv", "_temp.csv")
        
        return self.data_cleaner.run(
            input_path=input_path,
            output_path=cleaned_output_path,
            threshold=0.8,
            target_column=target_column
        )

    def _save_version_info(
        self,
        input_path: str,
        fe_output_path: str,
        final_output_path: str,
        new_transformations: List,
        all_transformations: List,
        description: str,
        final_dataset: pd.DataFrame,
        target_column: Optional[str],
        ml_score: float
    ):
        """Save all version-related information."""
        # Create version entry with ML score
        version_entry = self.version_manager.create_version_entry(
            input_path, fe_output_path, final_output_path,
            len(new_transformations), len(all_transformations), description,
            ml_score
        )
        
        # Save configuration
        config_path = self.version_manager.save_version_config(
            all_transformations, description, input_path,
            final_output_path, final_dataset, target_column
        )
        
        # Update version entry with config path
        version_entry["config_path"] = config_path

    def _build_final_result(self, all_transformations: List, final_dataset_path: str) -> Dict[str, Any]:
        """Build the final result dictionary."""
        return {
            'versions': self.version_manager.version_history,
            'final_dataset': final_dataset_path,
            'best_dataset': self.best_dataset_path,
            'transformations_count': len(all_transformations),
            'best_score': self.best_score,
            'best_version': self.best_version,
            'final_score': self.version_manager.version_history[-1].get('ml_score', 0.0) if self.version_manager.version_history else 0.0,
            'score_history': [v.get('ml_score', 0.0) for v in self.version_manager.version_history]
        }
