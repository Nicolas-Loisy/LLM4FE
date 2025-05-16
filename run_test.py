from pathlib import Path

from src.orchestrator.orchestrator import Orchestrator
from src.utils.logger import init_logger
from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline, Transformation

if __name__ == "__main__":
    fe_pipeline = FeatureEngineeringPipeline(dataset_path=Path("data/datasets/data.csv"), dataset_description="")
    fe_pipeline.transformations = [
        # Exemple math
        {
            "new_column_name": "example_final_col",
            "source_columns": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "add"}
        },
        # Exemple text - length
        {
            "new_column_name": "text_length",
            "source_columns": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "length"}
        },
        # Exemple text - word_count
        {
            "new_column_name": "text_word_count",
            "source_columns": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "word_count"}
        },
        {
            "new_column_name": "has_keyword",
            "source_columns": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "keyword", "keyword": "test"}
        },
        {
            "new_column_name": "tfidf",
            "source_columns": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "tfidf", "max_features": 1}
        },
        {
            "new_column_name": "date_column1_day",
            "source_columns": ["DateColumn1"],
            "transformation_type": "datetime_processing",
            "transformation_params": {"operation": "day"}
        },
        {
            "new_column_name": "days_diff",
            "source_columns": ["DateColumn1", "DateColumn2"],
            "transformation_type": "datetime_processing",
            "transformation_params": {"operation": "days_diff"}
        },
        {
            "new_column_name": "date_quarter",
            "source_columns": ["DateColumn1"],
            "transformation_type": "datetime_processing",
            "transformation_params": {
                "operation": "period",
                "freq": "Q"
            }
        }

    ]
    new_dataset = fe_pipeline.run()
    print(new_dataset.head())


    # print(Transformation.model_json_schema())