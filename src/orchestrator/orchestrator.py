import logging
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline
from src.data_cleanning.data_cleaner import DataCleaner
from src.orchestrator.version_manager import VersionManager
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config_path: str = "data/config.json"):
        """Initialize the orchestrator."""
        logger.info("Initializing Orchestrator...")
        self.config = get_config(config_path)
        self.version_manager = VersionManager()
        self.data_cleaner = DataCleaner()

    def run(
        self, 
        dataset_path: str, 
        dataset_description: Optional[str] = None, 
        target_column: Optional[str] = None,
        iterations: int = 1
    ) -> Dict[str, Any]:
        """Main entry point to run the feature engineering pipeline."""
        logger.info(f"Starting LLM4FE orchestration with {iterations} iterations...")
        
        try:
            current_dataset_path = dataset_path
            current_description = dataset_description
            all_transformations = []
            
            for i in range(iterations):
                logger.info(f"Starting iteration {i+1}/{iterations}...")
                
                result = self._run_single_iteration(
                    current_dataset_path, 
                    current_description, 
                    target_column, 
                    all_transformations
                )
                
                if result is None:
                    logger.error(f"Iteration {i+1} failed")
                    break
                
                current_dataset_path, current_description, all_transformations = result
                logger.info(f"Iteration {i+1} completed successfully")
            
            self.version_manager.save_global_summary()
            logger.info(f"Completed all {iterations} iterations")
            
            return self._build_final_result(current_dataset_path, all_transformations)
            
        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            raise

    def _run_single_iteration(
        self,
        dataset_path: str,
        description: Optional[str],
        target_column: Optional[str],
        all_transformations: List
    ) -> Optional[Tuple[str, str, List]]:
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
        
        # Update tracking
        updated_description = updated_description or description or ""
        all_transformations.extend(new_transformations)
        
        # Save version information
        self._save_version_info(
            dataset_path, fe_output_path, final_output_path,
            new_transformations, all_transformations, updated_description,
            final_dataset, target_column
        )
        
        return final_output_path, updated_description, all_transformations

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
        target_column: Optional[str]
    ):
        """Save all version-related information."""
        # Create version entry
        version_entry = self.version_manager.create_version_entry(
            input_path, fe_output_path, final_output_path,
            len(new_transformations), len(all_transformations), description
        )
        
        # Save configuration
        config_path = self.version_manager.save_version_config(
            all_transformations, description, input_path,
            final_output_path, final_dataset, target_column
        )
        
        # Update version entry with config path
        version_entry["config_path"] = config_path

    def _build_final_result(self, final_dataset_path: str, all_transformations: List) -> Dict[str, Any]:
        """Build the final result dictionary."""
        return {
            'versions': self.version_manager.version_history,
            'final_dataset': final_dataset_path,
            'transformations_count': len(all_transformations)
        }
