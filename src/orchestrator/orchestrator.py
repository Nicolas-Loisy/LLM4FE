import logging
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import os

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
        
        # Determine current prompt name for logging
        current_prompt_name = "default_prompt"
        for p_info in self.prompt_templates:
            if p_info['template'] == self.prompt_template:
                current_prompt_name = os.path.basename(p_info['name'])
                break
        
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
            iteration_scores = []  # Simple list of scores per iteration
            
            # Initialize with cleaned dataset score
            logger.info("Evaluating baseline cleaned dataset...")
            baseline_result = self._evaluate_dataset(current_dataset_path, target_column)
            baseline_score = baseline_result['score']
            self.problem_type = baseline_result['problem_type']
            self.best_score = baseline_score
            self.best_dataset_path = current_dataset_path
            self.best_version = 0
            
            # Record baseline
            iteration_scores.append({
                "iteration": 0,
                "prompt": current_prompt_name,
                "score": baseline_score,
                "is_best": True
            })
            
            logger.info(f"Baseline score (cleaned): {baseline_score:.4f}")
            
            iteration_count = 0
            consecutive_no_improvement = 0
            
            while iteration_count < max_iterations:
                iteration_count += 1
                logger.info(f"Starting iteration {iteration_count}/{max_iterations}...")
                
                result = self._run_single_iteration(
                    current_dataset_path, 
                    current_description, 
                    target_column, 
                    all_transformations,
                    iteration_count
                )
                
                if result is None:
                    logger.error(f"Iteration {iteration_count} failed")
                    iteration_scores.append({
                        "iteration": iteration_count,
                        "prompt": current_prompt_name,
                        "score": 0.0,
                        "is_best": False,
                        "status": "failed"
                    })
                    break
                
                fe_dataset_path, cleaned_dataset_path, current_description, all_transformations, ml_score = result
                
                # Use feature-engineered dataset for next iteration
                current_dataset_path = fe_dataset_path
                
                # Update best score tracking
                is_better, percentage_change = MachineLearningEstimator.get_best_score(ml_score, self.best_score, self.problem_type)
                
                if is_better:
                    logger.info(f"New best score found: {ml_score:.4f} (previous: {self.best_score:.4f}) - Improvement: {percentage_change:+.2f}%")
                    self.best_score = ml_score
                    self.best_dataset_path = cleaned_dataset_path
                    self.best_version = iteration_count
                    consecutive_no_improvement = 0
                else:
                    logger.info(f"Score {ml_score:.4f} did not improve best score {self.best_score:.4f} - Change: {percentage_change:+.2f}%")
                    consecutive_no_improvement += 1
                
                # Record iteration score
                iteration_scores.append({
                    "iteration": iteration_count,
                    "prompt": current_prompt_name,
                    "score": ml_score,
                    "is_best": is_better,
                    # "improvement": percentage_change if is_better else 0.0
                    "improvement": percentage_change
                })
                
                # Check stopping conditions
                should_stop = self._should_stop_iteration(
                    iteration_type, iteration_count, max_iterations, is_better, 
                    percentage_change, min_improvement_percentage, consecutive_no_improvement
                )
                
                if should_stop:
                    break
                
                logger.info(f"Iteration {iteration_count} completed successfully")
            
            logger.info(f"Completed {iteration_count} iterations.")
            logger.info(f"Best score: {self.best_score:.4f} from version {self.best_version}")
            
            return self._build_final_result(all_transformations, current_dataset_path, iteration_scores)
            
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
    ) -> Optional[Tuple[str, str, str, List, float]]:
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
        
        # Data Cleaning (only for evaluation)
        cleaned_output_path = self._run_data_cleaning(fe_output_path, target_column)
        final_dataset = pd.read_csv(cleaned_output_path)
        
        # ML Evaluation (using cleaned dataset)
        ml_evaluation_results = self._evaluate_dataset(cleaned_output_path, target_column)
        ml_score = ml_evaluation_results['score']
        logger.info(f"ML Score for iteration {iteration_number}: {ml_score:.4f}")
        
        # Update tracking
        updated_description = updated_description or description or ""
        all_transformations.extend(new_transformations)
        
        # Record iteration in version manager
        current_prompt_name = "default_prompt"
        for p_info in self.prompt_templates:
            if p_info['template'] == self.prompt_template:
                current_prompt_name = os.path.basename(p_info['name'])
                break
        
        self.version_manager.record_iteration(
            prompt_name=current_prompt_name,
            iteration=iteration_number,
            input_path=dataset_path,
            fe_output_path=fe_output_path,
            nb_transformations=len(new_transformations),
            dataset_description=updated_description,
            score=ml_score,
            transformations=new_transformations
        )
        
        return fe_output_path, cleaned_output_path, updated_description, all_transformations, ml_score

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

    def _build_final_result(self, all_transformations: List, final_dataset_path: str, iteration_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the final result dictionary."""
        return {
            'final_dataset': final_dataset_path,
            'best_dataset': self.best_dataset_path,
            'best_score': self.best_score,
            'best_version': self.best_version,
            'iteration_scores': iteration_scores,
            'total_iterations': len([s for s in iteration_scores if s['iteration'] > 0])
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
        """Run the orchestration with multiple prompts and compare results."""
        logger.info(f"Starting multi-prompt orchestration with {len(self.prompt_templates)} prompts...")
        
        if target_column is None:
            raise ValueError("target_column is required for ML evaluation")
        
        all_results = {}
        all_iteration_scores = []  # Collect all scores from all prompts
        global_best_score = -float('inf')
        global_best_prompt = None
        global_version_manager = VersionManager()  # Global version manager for all prompts
        
        for i, prompt_info in enumerate(self.prompt_templates):
            prompt_name = os.path.basename(prompt_info['name'])
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
                
                # Copy iteration data to global version manager
                prompt_iterations = self.version_manager.get_all_iterations()
                for iteration_data in prompt_iterations:
                    global_version_manager.iteration_history.append(iteration_data)
                
                # Store simplified result
                prompt_result = {
                    'prompt_name': prompt_name,
                    'best_score': result['best_score'],
                    'iteration_scores': result['iteration_scores'],
                    'total_iterations': result['total_iterations']
                }
                all_results[prompt_name] = prompt_result
                
                # Add to global iteration scores
                all_iteration_scores.extend(result['iteration_scores'])
                
                logger.info(f"Prompt '{prompt_name}' completed: Best Score {result['best_score']:.4f}")
                
                # Track global best
                if result['best_score'] > global_best_score:
                    global_best_score = result['best_score']
                    global_best_prompt = prompt_name
                
            except Exception as e:
                logger.error(f"Error running with prompt '{prompt_name}': {str(e)}")
                all_results[prompt_name] = {
                    'prompt_name': prompt_name,
                    'best_score': 0.0,
                    'iteration_scores': [],
                    'total_iterations': 0,
                    'error': str(e)
                }
        
        # Save global iterations summary
        summary_path = global_version_manager.save_iterations_summary()
        
        # Build final results
        final_results = {
            'global_best_score': global_best_score,
            'global_best_prompt': global_best_prompt,
            'all_iteration_scores': all_iteration_scores,
            'prompt_results': all_results,
            'summary': {
                name: {'best_score': result['best_score'], 'iterations': result['total_iterations']}
                for name, result in all_results.items()
            },
            'iterations_summary_path': summary_path
        }
        
        logger.info(f"\nGlobal Best: {global_best_prompt} with score {global_best_score:.4f}")
        logger.info(f"Iterations summary saved to: {summary_path}")
        
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
            if is_better and percentage_change < min_improvement_percentage:
                logger.info(f"Stopping iterations: Improvement {percentage_change:.2f}% is below threshold {min_improvement_percentage}%")
                return True
            elif not is_better:
                logger.info("Stopping iterations: No improvement in current iteration")
                return True
            return False
        
        return False
