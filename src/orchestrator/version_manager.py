import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class VersionManager:
    """Simplified version manager for basic dataset tracking with iteration logging."""
    
    def __init__(self, output_dir: str = "data/versions"):
        self.output_dir = output_dir
        self.current_version = 0
        self.iteration_history = []  # Store iteration data
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
    
    def increment_version(self) -> int:
        """Increment and return the current version number."""
        self.current_version += 1
        return self.current_version
    
    def save_dataset(self, dataset: pd.DataFrame, suffix: str = "") -> str:
        """Save dataset with versioned filename."""
        filename = f"dataset_v{self.current_version}{suffix}.csv"
        output_path = os.path.join(self.output_dir, filename)
        dataset.to_csv(output_path, index=False)
        logger.debug(f"Saved dataset to {output_path}")
        return output_path
    
    def record_iteration(
        self,
        prompt_name: str,
        iteration: int,
        input_path: str,
        fe_output_path: str,
        nb_transformations: int,
        dataset_description: str,
        score: float
    ) -> None:
        """Record iteration information."""
        iteration_data = {
            "version": self.current_version,
            "timestamp": datetime.now().isoformat(),
            "prompt_name": prompt_name,
            "iteration": iteration,
            "input_path": input_path,
            "fe_output_path": fe_output_path,
            "nb_transformations": nb_transformations,
            "dataset_description": dataset_description,
            "score": score
        }
        
        self.iteration_history.append(iteration_data)
        logger.debug(f"Recorded iteration {iteration} for prompt {prompt_name} with score {score:.4f}")
    
    def get_iterations_for_prompt(self, prompt_name: str) -> List[Dict[str, Any]]:
        """Get all iterations for a specific prompt."""
        return [item for item in self.iteration_history if item["prompt_name"] == prompt_name]
    
    def get_all_iterations(self) -> List[Dict[str, Any]]:
        """Get all recorded iterations."""
        return self.iteration_history.copy()
    
    def save_iterations_summary(self, filename: str = None) -> str:
        """Save all iteration data to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"all_prompts_iterations_{timestamp}.json"
        
        summary_path = os.path.join(self.output_dir, filename)
        
        summary_data = {
            "total_iterations": len(self.iteration_history),
            "prompts": list(set(item["prompt_name"] for item in self.iteration_history)),
            "iterations": self.iteration_history
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Saved iterations summary to {summary_path}")
        return summary_path
