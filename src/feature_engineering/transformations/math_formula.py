import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import re

from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class MathFormulaTransform(BaseTransformation):
    """
    Applies mathematical formulas to numeric columns using string expressions.
    """

    PROVIDER = "math_formula"
    DESCRIPTION = """
    This transformation applies mathematical formulas to numeric columns using string expressions.
    The formula is evaluated in a secure namespace containing column data and mathematical functions.

    Input:
        - source_columns: List of column names to use as variables in the formula.
        
    Output:
        - new_column_name: The name of the output column after applying the transformation.
        
    Param:
        - formula: A string containing the mathematical formula to apply.
                  Column names should be referenced by their exact names from source_columns in the formula.
                  DO NOT use generic references like 'col1', 'col2' - use the actual column names.
                  
                  Available operations and functions:
                  Basic operators: +, -, *, /, ** (power)
                  
                  Mathematical functions:
                  - sqrt(x): Square root
                  - log(x): Natural logarithm
                  - log10(x): Base-10 logarithm  
                  - log1p(x): log(1+x)
                  - exp(x): Exponential function
                  - abs(x): Absolute value
                  
                  Trigonometric functions:
                  - sin(x), cos(x), tan(x): Basic trigonometric functions
                  - asin(x), acos(x), atan(x): Inverse trigonometric functions
                  - sinh(x), cosh(x), tanh(x): Hyperbolic functions
                  
                  Rounding functions:
                  - floor(x): Round down to nearest integer
                  - ceil(x): Round up to nearest integer
                  - round(x): Round to nearest integer
                  
                  Array functions:
                  - max(a, b): Element-wise maximum
                  - min(a, b): Element-wise minimum
                  
                  Constants:
                  - pi: π (3.14159...)
                  - e: Euler's number (2.71828...)
                  - nan: Not a Number
                  - inf: Infinity
                  
                  Example formulas (assuming source_columns=['price', 'sqft', 'rooms']):
                  - "price**2 + 3*sqft - log(rooms)"
                  - "sqrt(price**2 + sqft**2)"  # Euclidean distance
                  - "sin(pi * price / 180)"      # Convert degrees to radians and get sine
                  - "max(price, sqft) / min(price, sqft)"  # Ratio of max to min
                  
        - safe_mode: (optional, default=True) If True, validates formula for security.
                    The formula is evaluated in a restricted namespace that only contains
                    the specified column data and mathematical functions. This prevents
                    access to potentially dangerous Python functions or modules.
        
    When safe_mode=True (recommended), the namespace is restricted to prevent security risks.
    The __builtins__ are disabled, and only the specified mathematical functions are available.
    """

    def __init__(self, new_column_name: str, source_columns: List[str], param: Optional[Dict[str, Any]] = None):
        """
        Initialize the math formula transformation.
        
        Args:
            new_column_name: The name of the output column after transformation
            source_columns: List of column names to use as variables in the formula
            param: Dictionary containing:
                - formula: Mathematical formula as string
                - safe_mode: (optional) Security validation flag
        """
        super().__init__(new_column_name, source_columns, param)
        
        # Validate param structure
        if not isinstance(param, dict) or "formula" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with a 'formula' key.")
        
        self.formula = param["formula"]
        self.safe_mode = param.get("safe_mode", True)
        
        # Validate formula safety if safe_mode is enabled
        if self.safe_mode:
            self._validate_formula_safety()
    
    def _validate_formula_safety(self):
        """
        Validate that the formula contains only safe mathematical operations.
        """
        # List of allowed functions and operators
        allowed_functions = [
            'sqrt', 'log', 'log10', 'log1p', 'exp', 'abs', 'sin', 'cos', 'tan',
            'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'floor', 'ceil',
            'round', 'max', 'min', 'sum', 'mean'
        ]
        
        # Remove allowed column names from formula for validation
        temp_formula = self.formula
        for col in self.source_columns:
            temp_formula = temp_formula.replace(col, 'X')
        
        # Remove numbers, operators, parentheses, and spaces
        temp_formula = re.sub(r'[0-9\+\-\*\/\(\)\s\.\,]', '', temp_formula)
        temp_formula = re.sub(r'\*\*', '', temp_formula)  # Remove power operator
        
        # Remove allowed functions
        for func in allowed_functions:
            temp_formula = temp_formula.replace(func, '')
        
        # Remove variable placeholder
        temp_formula = temp_formula.replace('X', '')
        
        # If anything remains, it's potentially unsafe
        if temp_formula.strip():
            raise ValueError(f"Formula contains potentially unsafe elements: {temp_formula}")
    
    def _prepare_namespace(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare the namespace for formula evaluation with column data and safe functions.
        """
        # Namespace Explanation:
        # The formula is evaluated using Python's eval() function within a controlled namespace.
        # This namespace contains:
        # 1. Column data: Each column in source_columns becomes a variable in the formula
        # 2. Mathematical functions: Safe numpy functions for mathematical operations
        # 3. Constants: Mathematical constants like pi and e
        namespace = {}
        
        # Add column data
        for col in self.source_columns:
            if col in df.columns:
                namespace[col] = df[col]
            else:
                logger.warning(f"Column '{col}' not found in DataFrame. Setting to NaN.")
                namespace[col] = pd.Series([np.nan] * len(df))
        
        # Add safe mathematical functions
        namespace.update({
            'sqrt': np.sqrt,
            'log': np.log,
            'log10': np.log10,
            'log1p': np.log1p,
            'exp': np.exp,
            'abs': np.abs,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'asin': np.arcsin,
            'acos': np.arccos,
            'atan': np.arctan,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
            'floor': np.floor,
            'ceil': np.ceil,
            'round': np.round,
            'max': np.maximum,
            'min': np.minimum,
            'pi': np.pi,
            'e': np.e,
            'nan': np.nan,
            'inf': np.inf
        })
        
        return namespace
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        
        try:
            # Prepare namespace with column data and functions
            namespace = self._prepare_namespace(df)
            
            # Evaluate the formula
            logger.debug(f"Evaluating formula: {self.formula}")
            
            # Use eval with restricted namespace for safety
            if self.safe_mode:
                # Restricted namespace for security
                result = eval(self.formula, {"__builtins__": {}}, namespace)
            else:
                # Full namespace (less secure but more flexible)
                result = eval(self.formula, namespace)
            
            # Ensure result is a pandas Series
            if not isinstance(result, pd.Series):
                if np.isscalar(result):
                    result = pd.Series([result] * len(df))
                else:
                    result = pd.Series(result)
            
            # Handle infinite values and very large numbers
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # Add the new column
            result_df[self.new_column_name] = result
            
            logger.debug(f"Successfully created column '{self.new_column_name}' using formula: {self.formula}")
            
        except ZeroDivisionError:
            logger.error("Division by zero encountered in formula evaluation")
            result_df[self.new_column_name] = np.nan
            
        except (ValueError, TypeError) as e:
            logger.error(f"Mathematical error in formula evaluation: {e}")
            result_df[self.new_column_name] = np.nan
            
        except Exception as e:
            logger.error(f"Unexpected error during formula transformation: {e}")
            result_df[self.new_column_name] = np.nan

        return result_df


# Example usage:
"""
# Create transformation instance
transform = MathFormulaTransform(
    new_column_name="price_per_sqft_ratio",
    source_columns=["price", "sqft", "rooms"],
    param={
        "formula": "price / sqft + log1p(rooms)",
        "safe_mode": True
    }
)

# Apply to DataFrame
df_transformed = transform.transform(df)
"""