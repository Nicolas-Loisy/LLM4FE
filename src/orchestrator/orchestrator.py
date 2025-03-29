# Orchestrator for managing the pipeline

import json
from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline
from src.automl.automl_pipeline import AutoMLPipeline
from src.benchmark.benchmark_pipeline import BenchmarkPipeline

# Orchestrator for managing the pipeline
class Orchestrator:
    def __init__(self):
        self.feature_engineering_pipeline = FeatureEngineeringPipeline()
        self.automl_pipeline = AutoMLPipeline()
        self.benchmark_pipeline = BenchmarkPipeline()

    def load_input_files(self):
        # Load input files like dataset.csv, config.json
        print("Loading input files...")

    def manage_versions(self):
        # Manage dataset and model versions
        print("Managing versions...")

    def run_feature_engineering(self):
        # Call the Feature Engineering module
        print("Running feature engineering...")
        self.feature_engineering_pipeline.generate_transformations()
        self.feature_engineering_pipeline.apply_transformations()
        self.feature_engineering_pipeline.save_transformed_dataset()

    def run_automl(self):
        # Call the AutoML module
        print("Running AutoML...")
        self.automl_pipeline.execute_pipeline()

    def run_benchmarking(self):
        # Call the Benchmarking module
        print("Running benchmarking...")
        self.benchmark_pipeline.execute_pipeline()

    def select_best_version(self):
        # Select the best model version based on scores
        print("Selecting best version...")


    def execute_pipeline(self):
        # Execute the entire pipeline
        self.load_input_files()
        self.manage_versions()
        self.run_feature_engineering()
        self.run_automl()
        self.run_benchmarking()
        self.select_best_version()
