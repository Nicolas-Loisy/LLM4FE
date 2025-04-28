from pathlib import Path

from src.orchestrator.orchestrator import Orchestrator
from src.utils.logger import init_logger
from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline, Transformation

if __name__ == "__main__":
    fe_pipeline = FeatureEngineeringPipeline(dataset_path=Path("data/datasets/data.csv"), dataset_description="")
    fe_pipeline.transformations = [
        {
            "new_column_name": "example_final_col",
            "source_columns": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "add"}
        }
    ]
    new_dataset = fe_pipeline.run()
    print(new_dataset.head())


    # print(Transformation.model_json_schema())