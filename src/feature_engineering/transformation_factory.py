from typing import Dict, Any, Optional, List
from src.feature_engineering.transformations.base_transformation import BaseTransformation

from src.feature_engineering.transformations.math_operations import MathOperationsTransform
from src.feature_engineering.transformations.encoding import EncodingTransform
from src.feature_engineering.transformations.scaling import ScalingTransform
from src.feature_engineering.transformations.text_processing import TextProcessingTransform
from src.feature_engineering.transformations.date_conversion import DateTimeProcessingTransform

class TransformationFactory:
    """
    Factory class for creating transformation instances based on configuration.

    """

    PROVIDER_TRANSFORMATIONS: List[str] = [
        MathOperationsTransform.PROVIDER,
        # EncodingTransform.PROVIDER,
        # ScalingTransform.PROVIDER,
        TextProcessingTransform.PROVIDER,
        # TODO : Add other transformation providers here
    ]

    # Informations about available transformations for llm prompt
    INFO_TRANSFORMATIONS = {
        MathOperationsTransform.PROVIDER: MathOperationsTransform.DESCRIPTION,
        # TODO : Add descriptions for other transformations
        # EncodingTransform.PROVIDER: EncodingTransform.DESCRIPTION,
        # ScalingTransform.PROVIDER: ScalingTransform.DESCRIPTION,
        TextProcessingTransform.PROVIDER: TextProcessingTransform.DESCRIPTION,
    }


    @staticmethod
    def create_transformation(transformation_config: Dict[str, Any]) -> Optional[BaseTransformation]:
        """
        Create a transformation based on the configuration.
        
        Args:
            transformation_config: Dictionary containing transformation configuration
                - new_column_name: Name of the column to store the result of the transformation
                - source_columns: List of columns to process
                - transformation_type: Type of transformation ('math', 'encoding', 'scaling', etc.)
                - transformation_params: Optional parameter for the transformation
                - ...
        Returns:
            A transformation instance or None if the provider is not found
        """
        provider = transformation_config["transformation_type"]
        param = transformation_config.get("transformation_params", None)
        new_column_name = transformation_config["new_column_name"]
        source_columns = transformation_config["source_columns"]

        if provider == "math_operations":
            return MathOperationsTransform(new_column_name, source_columns, param)

        if provider == "encoding":
            return EncodingTransform(new_column_name, source_columns, param)

        if provider == "scaling":
            return ScalingTransform(new_column_name, source_columns, param)

        if provider == "text_processing":
            return TextProcessingTransform(new_column_name, source_columns, param)

        if provider == "datetime_processing":
            return DateTimeProcessingTransform(new_column_name, source_columns, param)

        # Add more transformations as needed
        raise ValueError(f"Transformation provider '{provider}' not found.")
