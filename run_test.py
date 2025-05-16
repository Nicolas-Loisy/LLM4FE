from pathlib import Path
import pandas as pd

from src.orchestrator.orchestrator import Orchestrator
from src.utils.logger import init_logger
from src.feature_engineering.fe_pipeline import FeatureEngineeringPipeline, Transformation


if __name__ == "__main__":

    # Math transformation Test 
    fe_pipeline_1= FeatureEngineeringPipeline(dataset_path=Path("data/datasets/data.csv"), dataset_description="")
    fe_pipeline_1.transformations = [

        {
            "new_column_name": "multiply_final_col2",        
            "source_columns": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "multiply"}
        },
        {
            "new_column_name": "log_final_col3",
            "source_columns": ["Column1"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "log"}
        },
        {
            "new_column_name": "sqrt_final_col4",            
            "source_columns": ["Column1"],
            "transformation_type": "math_operations",   
            "transformation_params": {"operation": "sqrt"}
        },
        {
            "new_column_name": "square_final_col5",
            "source_columns": ["Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "square"}
        },          
        {
            "new_column_name": "mean_final_col6",
            "source_columns": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "mean"}
        },
        {
            "new_column_name": "sum_final_col7",
            "source_columns": ["Column1", "Column2"],
            "transformation_type": "math_operations",       
            "transformation_params": {"operation": "sum"}
        },
        {
            "new_column_name": "diff_final_col8",
            "source_columns": ["Column1", "Column2"],       
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "diff"}
        },
        {
            "new_column_name": "ratio_final_col9",
            "source_columns": ["Column1", "Column2"],
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "ratio"}
        },
        {
            "new_column_name": "addition_final_col10",
            "source_columns": ["Column1", "Column2"],   
            "transformation_type": "math_operations",
            "transformation_params": {"operation": "addition"}
        }
    ]
    new_dataset_1 = fe_pipeline_1.run()
    print(new_dataset_1.head())

    # Categorical transformation test 
    fe_pipeline = FeatureEngineeringPipeline(dataset_path=Path("data/datasets/categorical_data.csv"), dataset_description="")
    fe_pipeline.transformations = [
        {
            "new_column_name": "OneHot_encoding_Color",
            "source_columns": ["Color"],
            "transformation_type": "categorical_operations",
            "transformation_params": {"operation": "encodage_oneHot"}
        },
         {
            "new_column_name": "Label_encoding_Color",
            "source_columns": ["Color"],
            "transformation_type": "categorical_operations",
            "transformation_params": {"operation": "label_encoding"}
        },
         {
            "new_column_name": "Target_encoding_Color",
            "source_columns": ["Color"],
            "transformation_type": "categorical_operations",
            "transformation_params": {"operation": "target_encoding"}
        }

    ]
    new_dataset = fe_pipeline.run()
    print(new_dataset.head())

    #print(Transformation.model_json_schema())