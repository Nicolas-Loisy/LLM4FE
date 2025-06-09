from pathlib import Path
import pandas as pd

from src.utils.logger import init_logger
from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline, Transformation


if __name__ == "__main__":
    fe_pipeline_1 = FeatureEngineeringPipeline(dataset_path=Path("data/datasets/data.csv"), dataset_description="")
    fe_pipeline_1.transformations = [
        # Exemple math
        {
            "new_column_name": "multiply_final_col2",        
            "columns_to_process": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "multiply"}
        },
        {
            "new_column_name": "log_final_col3",
            "columns_to_process": ["Column1"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "log1p"}
        },
        {
            "new_column_name": "sqrt_final_col4",            
            "columns_to_process": ["Column1"],
            "transformation_type": "math_operations",   
            "transformation_params": {"operation": "sqrt"}
        },
        {
            "new_column_name": "square_final_col5",
            "columns_to_process": ["Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "square"}
        },          
        {
            "new_column_name": "mean_final_col6",
            "columns_to_process": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "mean"}
        },
        {
            "new_column_name": "sum_final_col7",
            "columns_to_process": ["Column1", "Column2"],
            "transformation_type": "math_operations",       
            "transformation_params": {"operation": "sum"}
        },
        {
            "new_column_name": "diff_final_col8",
            "columns_to_process": ["Column1", "Column2"],       
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "diff"}
        },
        {
            "new_column_name": "ratio_final_col9",
            "columns_to_process": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "ratio"}
        },
        {
            "new_column_name": "addition_final_col10",
            "columns_to_process": ["Column1", "Column2"],   
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "addition"}
        },
        # Exemple text - length
        {
            "new_column_name": "text_length",
            "columns_to_process": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "length"}
        },
        # Exemple text - word_count
        {
            "new_column_name": "text_word_count",
            "columns_to_process": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "word_count"}
        },
        {
            "new_column_name": "has_keyword",
            "columns_to_process": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "keyword", "keyword": "test"}
        },
        {
            "new_column_name": "tfidf",
            "columns_to_process": ["TextColumn"],
            "transformation_type": "text_processing",
            "transformation_params": {"operation": "tfidf", "max_features": 1}
        },
        {
            "new_column_name": "date_column1_day",
            "columns_to_process": ["DateColumn1"],
            "transformation_type": "datetime_processing",
            "transformation_params": {"operation": "day"}
        },
        {
            "new_column_name": "days_diff",
            "columns_to_process": ["DateColumn1", "DateColumn2"],
            "transformation_type": "datetime_processing",
            "transformation_params": {"operation": "days_diff"}
        },
        {
            "new_column_name": "date_quarter",
            "columns_to_process": ["DateColumn1"],
            "transformation_type": "datetime_processing",
            "transformation_params": {
                "operation": "period",
                "freq": "Q"
            }
        },
        {
            "new_column_name": "",
            "columns_to_process": ["DateColumn1"],
            "transformation_type": "delete",
            "transformation_params": {}
        },
    ]
    new_dataset_1 = fe_pipeline_1.run()
    print(new_dataset_1.head())

    # Categorical transformation test 
    fe_pipeline = FeatureEngineeringPipeline(dataset_path=Path("data/datasets/categorical_data.csv"), dataset_description="")
    fe_pipeline.transformations = [
        {
            "new_column_name": "OneHot_encoding_Color",
            "columns_to_process": ["Color"],
            "transformation_type": "categorical_operations",
            "transformation_params": {"operation": "encodage_oneHot"}
        },
         {
            "new_column_name": "Label_encoding_Color",
            "columns_to_process": ["Color"],
            "transformation_type": "categorical_operations",
            "transformation_params": {"operation": "label_encoding"}
        },
         {
            "new_column_name": "Target_encoding_Color",
            "columns_to_process": ["Color"],
            "transformation_type": "categorical_operations",
            "transformation_params": {"operation": "target_encoding"}
        }

    ]
    fe_pipeline.generate_transformations()
    new_dataset = fe_pipeline.run()
    print(new_dataset.head())

    # print(Transformation.model_json_schema())
