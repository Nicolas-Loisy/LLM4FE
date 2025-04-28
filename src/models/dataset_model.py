from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


# class NormalizationParams(BaseModel):
#     transformation_type: Literal['normalization']
#     method: Literal['minmax', 'zscore']
#     range_min: float
#     range_max: float

# class AggregationParams(BaseModel):
#     transformation_type: Literal['aggregation']
#     operation: Literal['sum', 'mean', 'max', 'min']
#     window_size: int

# class EncodingParams(BaseModel):
#     transformation_type: Literal['encoding']
#     encoding_type: Literal['onehot', 'label']

# Define a type for transformation parameters
TransformationParams = Union[
    # NormalizationParams, 
    # AggregationParams, 
    # EncodingParams, 
    Dict[str, str | int]
]

class ColumnStructure(BaseModel):
    new_column_name: str = Field(..., description="The new column name")
    column_description: str = Field(..., description="The description of the column")
    source_columns: List[str] = Field(..., description="List of source column names")
    transformation_type: str = Field(..., description="Type of transformation applied")
    # Updated to use specific transformation parameter models
    transformation_params: Optional[TransformationParams] = Field(default_factory=dict, description="Parameters for transformation")

class DatasetStructure(BaseModel):
    dataset_structure: List[ColumnStructure] = Field(..., description="List of column structures")
