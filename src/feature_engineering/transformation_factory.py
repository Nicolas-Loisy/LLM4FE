from typing import Dict, Any, Optional, List
from src.feature_engineering.transformations.base_transformation import BaseTransformation
from src.feature_engineering.transformations.math_operations import MathOperationsTransform
from src.feature_engineering.transformations.text_processing import TextProcessingTransform
from src.feature_engineering.transformations.categorical_operations import CategoricalOperationsTransform
from src.feature_engineering.transformations.datetime_processing import DateTimeProcessingTransform
from src.feature_engineering.transformations.delete_column import DeleteColumnTransform
from src.feature_engineering.transformations.math_formula import MathFormulaTransform

class TransformationFactory:
    """
    Factory class for creating transformation instances based on configuration.

    """

    PROVIDER_TRANSFORMATIONS: List[str] = [
        MathOperationsTransform.PROVIDER,
        TextProcessingTransform.PROVIDER,
        DeleteColumnTransform.PROVIDER,
        CategoricalOperationsTransform.PROVIDER,
        DateTimeProcessingTransform.PROVIDER,
        MathFormulaTransform.PROVIDER,
        # TODO : Add other transformation providers here
    ]

    # Informations about available transformations for llm prompt
    INFO_TRANSFORMATIONS = {
        MathOperationsTransform.PROVIDER: MathOperationsTransform.DESCRIPTION,
        TextProcessingTransform.PROVIDER: TextProcessingTransform.DESCRIPTION,
        DeleteColumnTransform.PROVIDER: DeleteColumnTransform.DESCRIPTION,
        CategoricalOperationsTransform.PROVIDER: CategoricalOperationsTransform.DESCRIPTION,
        DateTimeProcessingTransform.PROVIDER: DateTimeProcessingTransform.DESCRIPTION,
        MathFormulaTransform.PROVIDER: MathFormulaTransform.DESCRIPTION,
        # TODO : Add descriptions for other transformations
    }


    @staticmethod
    def create_transformation(transformation_config: Dict[str, Any]) -> Optional[BaseTransformation]:
        """
        Create a transformation based on the configuration.
        
        Args:
            transformation_config: Dictionary containing transformation configuration
                - new_column_name: Name of the column to store the result of the transformation
                - columns_to_process: List of columns to process
                - transformation_type: Type of transformation ('math', etc.)
                - transformation_params: Optional parameter for the transformation
                - ...
        Returns:
            A transformation instance or None if the provider is not found
        """
        provider = transformation_config.get("provider_transform", None)
        param = transformation_config.get("params", None)
        new_column_name = transformation_config.get("new_column_name", None)
        columns_to_process = transformation_config.get("columns_to_process", [])

        if provider == MathOperationsTransform.PROVIDER:
            return MathOperationsTransform(new_column_name, columns_to_process, param)

        if provider == TextProcessingTransform.PROVIDER:
            return TextProcessingTransform(new_column_name, columns_to_process, param)

        if provider == CategoricalOperationsTransform.PROVIDER:
            return CategoricalOperationsTransform(new_column_name, columns_to_process, param)

        if provider == DateTimeProcessingTransform.PROVIDER:
            return DateTimeProcessingTransform(new_column_name, columns_to_process, param)

        if provider == DeleteColumnTransform.PROVIDER:
            return DeleteColumnTransform(new_column_name, columns_to_process, param)

        if provider == MathFormulaTransform.PROVIDER:
            return MathFormulaTransform(new_column_name, columns_to_process, param)

        # Add more transformations as needed
        raise ValueError(f"Transformation provider '{provider}' not found.")
