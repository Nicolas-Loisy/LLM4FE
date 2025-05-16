from pathlib import Path

from src.utils.logger import init_logger
from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline, Transformation

if __name__ == "__main__":
    logging_path = Path(__file__).parent / "logging.ini"
    init_logger(logger_path=logging_path)

    fe_pipeline = FeatureEngineeringPipeline(
        dataset_path=Path("data/datasets/data.csv"), 
    )

    # Example transformation configuration
    fe_pipeline.transformations = [
        {
            "new_column_name": "example_final_col",
            "source_columns": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "add"}
        }
    ]
    fe_pipeline.generate_transformations()
    new_dataset = fe_pipeline.run()
    print(new_dataset.head())

    # print(Transformation.model_json_schema())