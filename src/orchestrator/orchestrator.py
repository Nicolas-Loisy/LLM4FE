# Orchestrator for managing the pipeline
import json
import os
import logging

from datetime import datetime
from typing import Dict, Any, Optional

# from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline
# from src.automl.automl_pipeline import AutoMLPipeline
# from src.benchmark.benchmark_pipeline import BenchmarkPipeline
# from src.utils.config import Config

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, config_path: str = "config.json", dataset_description: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config_path: Path to the configuration file
            dataset_description: Optional description of the dataset for feature engineering
        """

        logger.info("Initializing Orchestrator...")

        # self.config = Config(config_path)
        # self.feature_engineering_pipeline = FeatureEngineeringPipeline(dataset_description=dataset_description)
        # # self.automl_pipeline = AutoMLPipeline()
        # # self.benchmark_pipeline = BenchmarkPipeline()
        
        # self.current_version = 1
        # self.versions_info = {}
        # self.input_dataset_path = None
        # self.output_dir = "data"
        # self.models_dir = os.path.join(self.output_dir, "models")
        # self.logs_dir = os.path.join(self.output_dir, "logs")
        
        # # Create directories if they don't exist
        # os.makedirs(self.models_dir, exist_ok=True)
        # os.makedirs(self.logs_dir, exist_ok=True)

    def manage_versions(self):
        """
        Manage dataset and model versions.
        """
        print("Managing versions...")
        
        # Create a new version directory
        version_dir = os.path.join(self.models_dir, f"version_{self.current_version}")
        os.makedirs(version_dir, exist_ok=True)
        
        # Update the version in the feature engineering pipeline
        self.feature_engineering_pipeline.set_version(self.current_version)
        
        # Store version information
        self.versions_info[self.current_version] = {
            "version": self.current_version,
            "timestamp": datetime.now().isoformat(),
            "input_dataset": self.input_dataset_path,
            "version_dir": version_dir
        }
        
        print(f"Created version directory: {version_dir}")
        return version_dir

    def run_feature_engineering(self) -> str:
        """
        Call the Feature Engineering module using a single run method.
        
        Returns:
            Path to the transformed dataset
        """
        print(f"Running feature engineering for version {self.current_version}...")
        
        # Set the version in the feature engineering pipeline
        self.feature_engineering_pipeline.set_version(self.current_version)
        
        # Get the version directory
        version_dir = self.versions_info[self.current_version]["version_dir"]
        
        # Run the feature engineering pipeline
        result = self.feature_engineering_pipeline.run(
            dataset_path=self.input_dataset_path, 
            output_dir=version_dir
        )
        
        if result["status"] == "success":
            # Update version information
            self.versions_info[self.current_version]["transformed_dataset"] = result["output_dataset"]
            self.versions_info[self.current_version]["transformations"] = result["transformations"]
            
            # Save transformations to file
            transformations_path = os.path.join(version_dir, f"transformations_v{self.current_version}.json")
            with open(transformations_path, 'w') as f:
                json.dump({"transformations": result["transformations"]}, f, indent=4)
            
            print(f"Feature engineering completed. Transformed dataset saved to {result['output_dataset']}")
            return result["output_dataset"]
        else:
            print(f"Feature engineering failed: {result.get('message', 'Unknown error')}")
            return None

    def run_automl(self) -> str:
        """
        Call the AutoML module.
        
        Returns:
            Path to the trained model
        """
        print(f"Running AutoML for version {self.current_version}...")
        
        # Get the transformed dataset path
        transformed_dataset_path = self.versions_info[self.current_version]["transformed_dataset"]
        
        # Set the version directory
        version_dir = self.versions_info[self.current_version]["version_dir"]
        
        # Run the AutoML pipeline
        model_path = self.automl_pipeline.execute_pipeline(
            dataset_path=transformed_dataset_path,
            output_dir=version_dir
        )
        
        # Update version information
        self.versions_info[self.current_version]["model"] = model_path
        
        print(f"AutoML completed. Model saved to {model_path}")
        return model_path

    def run_benchmarking(self) -> Dict[str, Any]:
        """
        Call the Benchmarking module.
        
        Returns:
            Dictionary of benchmark scores
        """
        print(f"Running benchmarking for version {self.current_version}...")
        
        # Get the model path and dataset path
        model_path = self.versions_info[self.current_version]["model"]
        dataset_path = self.versions_info[self.current_version]["transformed_dataset"]
        
        # Set the version directory
        version_dir = self.versions_info[self.current_version]["version_dir"]
        
        # Run the benchmarking pipeline
        scores = self.benchmark_pipeline.execute_pipeline(
            model_path=model_path,
            dataset_path=dataset_path,
            output_dir=version_dir
        )
        
        # Update version information
        self.versions_info[self.current_version]["scores"] = scores
        
        # Save scores to file
        scores_path = os.path.join(version_dir, f"Model_Scores_v{self.current_version}.json")
        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=4)
        
        print(f"Benchmarking completed. Scores saved to {scores_path}")
        return scores

    def select_best_version(self) -> int:
        """
        Select the best model version based on scores.
        
        Returns:
            The version number of the best model
        """
        print("Selecting best version...")
        
        best_version = None
        best_score = -float('inf')
        
        # Metric to use for comparison (e.g., 'accuracy', 'f1', etc.)
        metric = self.config.config.get("evaluation_metric", "accuracy")
        
        for version, info in self.versions_info.items():
            if "scores" in info and metric in info["scores"]:
                score = info["scores"][metric]
                
                if score > best_score:
                    best_score = score
                    best_version = version
        
        if best_version is not None:
            print(f"Best version: {best_version} with {metric} = {best_score}")
            
            # Update the configuration with the best version
            self.config.config["best_version"] = best_version
            self.config.config["best_score"] = best_score
            self.config.save_config()
            
            return best_version
        else:
            print("No versions with scores found.")
            return None

    def iterate_pipeline(self, iterations: int = 1):
        """
        Run multiple iterations of the pipeline.
        
        Args:
            iterations: Number of iterations to run
        """
        print(f"Running {iterations} iterations of the pipeline...")
        
        for i in range(iterations):
            print(f"\n--- Iteration {i+1}/{iterations} ---\n")
            
            # Run the pipeline for the current version
            self.execute_pipeline()
            
            # Increment the version for the next iteration
            self.current_version += 1
            
            # If this is not the last iteration, use the transformed dataset as input
            if i < iterations - 1:
                # Get the transformed dataset from the previous version
                prev_transformed_dataset = self.versions_info[self.current_version - 1]["transformed_dataset"]
                
                # Update the input dataset for the next iteration
                self.feature_engineering_pipeline.load_dataset(prev_transformed_dataset)
        
        # Select the best version after all iterations
        best_version = self.select_best_version()
        
        print(f"\nPipeline completed with {iterations} iterations.")
        print(f"Best version: {best_version}")
        
        return best_version

    def execute_pipeline(self):
        """
        Execute the entire pipeline for a single version.
        """
        print(f"\n=== Executing pipeline for version {self.current_version} ===\n")
        
        # Manage versions
        version_dir = self.manage_versions()
        
        # Run feature engineering
        transformed_dataset_path = self.run_feature_engineering()
        
        # Run AutoML
        model_path = self.run_automl()
        
        # Run benchmarking
        scores = self.run_benchmarking()
        
        print(f"\n=== Pipeline execution completed for version {self.current_version} ===\n")
        
        # Save the version information
        version_info_path = os.path.join(version_dir, f"version_info_v{self.current_version}.json")
        with open(version_info_path, 'w') as f:
            # Create a serializable copy of the version info
            version_info = self.versions_info[self.current_version].copy()
            # Convert any non-serializable objects to strings
            for key, value in version_info.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    version_info[key] = str(value)
            
            json.dump(version_info, f, indent=4)
        
        return self.versions_info[self.current_version]

    def run(self, dataset_path: str, iterations: int = 1) -> Dict[str, Any]:
        """
        Main entry point to run the complete orchestration pipeline.
        
        Args:
            dataset_path: Path to the input dataset
            iterations: Number of pipeline iterations to run
            
        Returns:
            Dictionary containing information about the best model version
        """
        print("Starting LLM4FE orchestration pipeline...")
        
        # Store the input dataset path
        self.input_dataset_path = dataset_path
        
        # Run the pipeline iterations
        best_version = self.iterate_pipeline(iterations=iterations)
        
        # Return the information about the best version
        if best_version is not None:
            return self.versions_info[best_version]
        else:
            return {"status": "completed", "message": "No best version identified"}
