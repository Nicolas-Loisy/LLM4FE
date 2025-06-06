import logging
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline
from src.data_cleanning.data_cleaner import DataCleaner
from src.orchestrator.version_manager import VersionManager
from src.ml_generator.ml_estimator import MachineLearningEstimator
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class IterationType(Enum):
    """
    Enum to represent the type of iteration control for the orchestrator.

    - FIXED: Run a fixed number of iterations.
    - SCORE_IMPROVEMENT: Stop if there is no improvement in score.
    - PERCENTAGE_IMPROVEMENT: Stop if the improvement is below a minimum percentage.
    """
    FIXED = "fixed"
    SCORE_IMPROVEMENT = "score_improvement" 
    PERCENTAGE_IMPROVEMENT = "percentage_improvement"


class Orchestrator:
    def __init__(self, config_path: str = "data/configs/config.json"):
        """Initialize the orchestrator."""
        logger.info("Initializing Orchestrator...")
        self.config = get_config(config_path)
        self.version_manager = VersionManager() # Définir le dossier de save des versions
        self.data_cleaner = DataCleaner()
        self.best_score = -float('inf')
        self.best_dataset_path = None
        self.best_version = None
        self.problem_type = None  # Track ML problem type
        
        # Load the default prompt template
        self.prompt_template = self.config.get_file_content("prompt_file")
        if not self.prompt_template:
            raise ValueError("Failed to load prompt template from config")
        logger.info("Default prompt template loaded successfully")
        
        # Load multiple prompt templates if available
        self.prompt_templates = []
        prompt_files = self.config.get("prompt_files", [])
        if prompt_files:
            for prompt_file in prompt_files:
                template = self.config.get_file_content_by_path(prompt_file)
                if template:
                    self.prompt_templates.append({
                        'name': prompt_file,
                        'template': template
                    })
                    logger.info(f"Loaded prompt template: {prompt_file}")
                else:
                    logger.warning(f"Failed to load prompt template: {prompt_file}")
        
        if not self.prompt_templates:
            # Fallback to single prompt
            self.prompt_templates = [{
                'name': self.config.get("prompt_file", "default"),
                'template': self.prompt_template
            }]

    def run(
        self, 
        dataset_path: str, 
        dataset_description: Optional[str] = None, 
        target_column: Optional[str] = None,
        max_iterations: int = 3,
        iteration_type: IterationType = IterationType.FIXED,
        min_improvement_percentage: float = 1.0
    ) -> Dict[str, Any]:
        """Main entry point to run the feature engineering pipeline."""
        logger.info(f"Starting LLM4FE orchestration with iteration type: {iteration_type.value}")
        
        if iteration_type == IterationType.FIXED:
            logger.info(f"Fixed iterations: {max_iterations}")
        elif iteration_type == IterationType.SCORE_IMPROVEMENT:
            logger.info(f"Score improvement mode (max iterations: {max_iterations})")
        elif iteration_type == IterationType.PERCENTAGE_IMPROVEMENT:
            logger.info(f"Percentage improvement mode: {min_improvement_percentage}% (max iterations: {max_iterations})")
        
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
            baseline_result = self._evaluate_dataset(current_dataset_path, target_column)
            baseline_score = baseline_result['score']
            self.problem_type = baseline_result['problem_type']
            self.best_score = baseline_score
            self.best_dataset_path = current_dataset_path
            self.best_version = 0
            
            logger.info(f"Baseline score (cleaned): {baseline_score:.4f}")
            
            iteration_count = 0
            consecutive_no_improvement = 0
            
            while iteration_count < max_iterations:
                iteration_count += 1
                logger.info(f"Starting iteration {iteration_count}/{max_iterations}...")
                logger.info(f"Using dataset: {current_dataset_path}")
                
                result = self._run_single_iteration(
                    current_dataset_path, 
                    current_description, 
                    target_column, 
                    all_transformations,
                    iteration_count
                )
                
                if result is None:
                    logger.error(f"Iteration {iteration_count} failed")
                    break
                
                iteration_dataset_path, current_description, all_transformations, ml_score = result
                
                # Always use the last generated dataset for next iteration
                current_dataset_path = iteration_dataset_path
                
                # Update best score tracking using get_best_score
                is_better, percentage_change = MachineLearningEstimator.get_best_score(ml_score, self.best_score, self.problem_type)
                
                if is_better:
                    logger.info(f"New best score found: {ml_score:.4f} (previous: {self.best_score:.4f}) - Improvement: {percentage_change:+.2f}%")
                    self.best_score = ml_score
                    self.best_dataset_path = iteration_dataset_path
                    self.best_version = iteration_count
                    consecutive_no_improvement = 0
                else:
                    logger.info(f"Score {ml_score:.4f} did not improve best score {self.best_score:.4f} - Change: {percentage_change:+.2f}%")
                    consecutive_no_improvement += 1
                
                # Check stopping conditions
                should_stop = self._should_stop_iteration(
                    iteration_type, iteration_count, max_iterations, is_better, 
                    percentage_change, min_improvement_percentage, consecutive_no_improvement
                )
                
                if should_stop:
                    break
                else:
                    logger.info(f"Continuing with last generated dataset for next iteration")
                
                logger.info(f"Iteration {iteration_count} completed successfully")
            
            self.version_manager.save_global_summary()
            logger.info(f"Completed {iteration_count} iterations.")
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
        ml_score = self._evaluate_dataset(final_output_path, target_column)['score']
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

    def _evaluate_dataset(self, dataset_path: str, target_column: str) -> Dict[str, Any]:
        """Evaluate dataset using ML estimator and return score and detailed results."""
        try:
            ml_estimator = MachineLearningEstimator(
                dataset_path=dataset_path,
                target_col=target_column
            )
            results = ml_estimator.run()
            return {
                'score': results.get('score', 0.0),
                'problem_type': results.get('problem_type', 'unknown'),
                'metric_name': results.get('metric_name', 'unknown')
            }
        except Exception as e:
            logger.error(f"ML evaluation failed: {str(e)}")
            return {'score': 0.0, 'problem_type': 'unknown', 'metric_name': 'unknown'}

    def _run_feature_engineering(
        self, 
        dataset_path: str, 
        description: Optional[str], 
        target_column: Optional[str]
    ) -> Tuple[Optional[pd.DataFrame], List, Optional[str]]:
        """Execute feature engineering pipeline."""
        fe_pipeline = FeatureEngineeringPipeline(
            dataset_path=dataset_path,
            prompt=self.prompt_template,
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

    def run_multiple_prompts(
        self,
        dataset_path: str,
        dataset_description: Optional[str] = None,
        target_column: Optional[str] = None,
        max_iterations: int = 3,
        iteration_type: IterationType = IterationType.FIXED,
        min_improvement_percentage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run the orchestration with multiple prompts and compare results.
        
        Args:
            dataset_path: Path to the input dataset
            dataset_description: Optional description of the dataset
            target_column: Target column for ML evaluation
            max_iterations: Maximum number of iterations to run
            iteration_type: Type of iteration control (fixed, score improvement, percentage improvement)
            min_improvement_percentage: Minimum improvement percentage for stopping criteria
            
        Returns:
            Dictionary with results for each prompt and overall best results
        """
        logger.info(f"Starting multi-prompt orchestration with {len(self.prompt_templates)} prompts...")
        logger.info(f"Iteration type: {iteration_type.value}")
        
        if target_column is None:
            raise ValueError("target_column is required for ML evaluation")
        
        all_results = {}
        global_best_score = -float('inf')
        global_best_prompt = None
        global_best_result = None
        
        for i, prompt_info in enumerate(self.prompt_templates):
            prompt_name = prompt_info['name']
            prompt_template = prompt_info['template']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING WITH PROMPT {i+1}/{len(self.prompt_templates)}: {prompt_name}")
            logger.info(f"{'='*60}")
            
            try:
                # Reset state for each prompt
                self._reset_state()
                
                # Set current prompt template
                self.prompt_template = prompt_template
                
                # Run orchestration with current prompt
                result = self.run(
                    dataset_path=dataset_path,
                    dataset_description=dataset_description,
                    target_column=target_column,
                    max_iterations=max_iterations,
                    iteration_type=iteration_type,
                    min_improvement_percentage=min_improvement_percentage
                )
                
                # Store result with prompt info
                result['prompt_name'] = prompt_name
                result['prompt_template'] = prompt_template
                all_results[prompt_name] = result
                
                logger.info(f"Prompt '{prompt_name}' completed:")
                logger.info(f"  Best Score: {result['best_score']:.4f}")
                logger.info(f"  Final Score: {result['final_score']:.4f}")
                logger.info(f"  Transformations: {result['transformations_count']}")
                
                # Track global best
                if result['best_score'] > global_best_score:
                    global_best_score = result['best_score']
                    global_best_prompt = prompt_name
                    global_best_result = result
                    logger.info(f"New global best score: {global_best_score:.4f} with prompt '{prompt_name}'")
                
            except Exception as e:
                logger.error(f"Error running with prompt '{prompt_name}': {str(e)}")
                all_results[prompt_name] = {
                    'error': str(e),
                    'prompt_name': prompt_name,
                    'best_score': 0.0,
                    'final_score': 0.0
                }
        
        # Build comprehensive results
        final_results = {
            'prompt_results': all_results,
            'global_best_prompt': global_best_prompt,
            'global_best_score': global_best_score,
            'global_best_result': global_best_result,
            'prompts_compared': len(self.prompt_templates),
            'prompt_summary': {
                name: {
                    'best_score': result.get('best_score', 0.0),
                    'final_score': result.get('final_score', 0.0),
                    'transformations_count': result.get('transformations_count', 0)
                }
                for name, result in all_results.items()
            }
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("MULTI-PROMPT ORCHESTRATION COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Global Best Prompt: {global_best_prompt}")
        logger.info(f"Global Best Score: {global_best_score:.4f}")
        
        return final_results
    
    def _reset_state(self):
        """Reset orchestrator state for new prompt execution."""
        self.best_score = -float('inf')
        self.best_dataset_path = None
        self.best_version = None
        self.problem_type = None
        # Create new version manager for each prompt run
        self.version_manager = VersionManager()

    def _should_stop_iteration(
        self, 
        iteration_type: IterationType, 
        current_iteration: int, 
        max_iterations: int,
        is_better: bool, 
        percentage_change: float, 
        min_improvement_percentage: float,
        consecutive_no_improvement: int
    ) -> bool:
        """Determine if iterations should stop based on the iteration type and conditions."""
        
        if iteration_type == IterationType.FIXED:
            # For fixed iterations, continue until max reached
            return False
        
        elif iteration_type == IterationType.SCORE_IMPROVEMENT:
            # Stop if no improvement in the last iteration
            if not is_better:
                logger.info("Stopping iterations: No score improvement in current iteration")
                return True
            return False
        
        elif iteration_type == IterationType.PERCENTAGE_IMPROVEMENT:
            # Stop if improvement percentage is below threshold
            if is_better and abs(percentage_change) < min_improvement_percentage:
                logger.info(f"Stopping iterations: Improvement {percentage_change:.2f}% is below threshold {min_improvement_percentage}%")
                return True
            elif not is_better:
                logger.info("Stopping iterations: No improvement in current iteration")
                return True
            return False
        
        return False
