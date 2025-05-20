from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field

from src.feature_engineering.transformation_factory import TransformationFactory


class Transformation(BaseModel):
    """
    Pydantic Model representing a single transformation
    """
    final_col: str = Field(..., description="The name of the resulting column after transformation")
    cols_to_process: List[str] = Field(..., description="List of source columns to process")
    provider_transform: Literal[*TransformationFactory.PROVIDER_TRANSFORMATIONS] = Field(..., description="The transformation provider to use")
    params: Optional[Dict[str, Union[str, int, float, bool, None]]] = Field(None, description="Optional parameters for the transformation")

class DatasetStructure(BaseModel):
    """
    Pydantic Model representing the structure of the dataset modifications.
    """
    dataset_description: Optional[str] = Field(None, description="Description of the dataset with a brief description of each column")
    datasetStructure: List[Transformation] = Field(..., description="List of transformations applied to the dataset")
