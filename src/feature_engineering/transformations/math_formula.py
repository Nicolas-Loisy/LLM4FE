import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import re
import difflib
from src.feature_engineering.transformations.base_transformation import BaseTransformation

logger = logging.getLogger(__name__)

class MathFormulaTransform(BaseTransformation):
    """
    Applies a mathematical formula to numeric columns using a Python expression evaluated vectorially on the DataFrame.
    """

    PROVIDER = "math_formula"
    DESCRIPTION = """
    Applies a mathematical formula to one or more numeric columns using a Python expression evaluated vectorially on the DataFrame.

    - columns_to_process: list of exact column names to use as variables in the formula.
    - new_column_name: name of the output column.
    - param:
        - formula: a Python expression evaluated vectorially.
        Allowed functions: +, -, *, /, **, np.sqrt, np.log, np.log10, np.exp, np.abs, np.sin, np.cos, np.tan,
        np.arcsin, np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.floor, np.ceil, np.round,
        np.maximum, np.minimum, np.pi, np.e, np.inf, np.nan, np.where
        Example: "np.where(column1 != 0, column2 / column1, 0)"

    IMPORTANT:
    - The formula must use **only the exact column names** from `columns_to_process` as variables.
    - For mathematical functions, always use the 'np.' prefix, e.g., 'np.sqrt(column_name)'.
    - Do NOT create intermediate variables or aliases (like 'sqrt_column_name').
    - The expression is evaluated vectorially on the entire DataFrame.
    """

    def __init__(self, new_column_name: str, columns_to_process: List[str], param: Optional[Dict[str, Any]] = None):
        super().__init__(new_column_name, columns_to_process, param)

        if not isinstance(param, dict) or "formula" not in param:
            raise ValueError("Invalid param structure. Expected a dictionary with a 'formula' key.")

        # Correct formula columns automatically
        original_formula = param["formula"]
        self.formula = self.autocorrect_formula_columns(self.columns_to_process, original_formula)

        self.safe_mode = param.get("safe_mode", False)

        if self.safe_mode:
            self._validate_formula_safety()

    def autocorrect_formula_columns(self, columns_to_process: List[str], formula: str) -> str:
        """
        Automatically correct column names in the formula based on columns_to_process.
        It replaces any token that is not in columns_to_process and looks similar by the closest match.

        Args:
            columns_to_process: List of exact column names expected in the formula.
            formula: The formula string potentially containing wrong column names.

        Returns:
            Corrected formula string with column names fixed.
        """
        tokens = set(re.findall(r'\b\w+\b', formula))

        np_funcs = {
            'np', 'sqrt', 'log', 'log10', 'exp', 'abs', 'sin', 'cos', 'tan',
            'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'floor', 'ceil',
            'round', 'maximum', 'minimum', 'pi', 'e', 'inf', 'nan', 'where'
        }
        python_keywords = {
            'if', 'else', 'for', 'and', 'or', 'not', 'in', 'True', 'False', 'None'
        }

        candidates = [tok for tok in tokens
                      if tok not in columns_to_process
                      and tok not in np_funcs
                      and tok not in python_keywords
                      and not tok.isdigit()]

        corrected_formula = formula
        for candidate in candidates:
            matches = difflib.get_close_matches(candidate, columns_to_process, n=1, cutoff=0.6)
            if matches:
                corrected_name = matches[0]
                corrected_formula = re.sub(rf'\b{re.escape(candidate)}\b', corrected_name, corrected_formula)

        if corrected_formula != formula:
            logger.info(f"Formula corrected from:\n  {formula}\nto:\n  {corrected_formula}")

        return corrected_formula

    def _validate_formula_safety(self):
        allowed_tokens = [
            'np.sqrt', 'np.log', 'np.log10', 'np.exp', 'np.abs', 'np.sin', 'np.cos', 'np.tan',
            'np.arcsin', 'np.arccos', 'np.arctan', 'np.sinh', 'np.cosh', 'np.tanh', 'np.floor', 'np.ceil',
            'np.round', 'np.maximum', 'np.minimum', 'np.pi', 'np.e', 'np.inf', 'np.nan', 'np.where',
            '+', '-', '*', '/', '**', '(', ')', ',', '.', ' '
        ]

        temp_formula = self.formula
        for col in self.columns_to_process:
            temp_formula = temp_formula.replace(col, 'X')

        for token in allowed_tokens:
            temp_formula = temp_formula.replace(token, '')

        temp_formula = re.sub(r'[0-9]', '', temp_formula)

        if temp_formula.strip():
            raise ValueError(f"Formula contains potentially unsafe elements: {temp_formula}")

    def _prepare_namespace(self, df: pd.DataFrame) -> Dict[str, Any]:
        namespace = {}
        for col in self.columns_to_process:
            if col in df.columns:
                namespace[col] = df[col]
            else:
                logger.warning(f"Column '{col}' not found in DataFrame. Filling with NaN.")
                namespace[col] = pd.Series([np.nan] * len(df))

        namespace['np'] = np
        return namespace

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        try:
            namespace = self._prepare_namespace(df)
            logger.debug(f"Evaluating formula vectorially: {self.formula}")

            if self.safe_mode:
                result = eval(self.formula, {"__builtins__": {}}, namespace)
            else:
                result = eval(self.formula, {}, namespace)

            if not isinstance(result, pd.Series):
                if np.isscalar(result):
                    result = pd.Series([result] * len(df))
                else:
                    result = pd.Series(result)

            result = result.replace([np.inf, -np.inf], np.nan)
            result_df[self.new_column_name] = result
            logger.debug(f"Column '{self.new_column_name}' created successfully.")

        except ZeroDivisionError:
            logger.error("Division by zero encountered during formula evaluation.")
            result_df[self.new_column_name] = np.nan

        except Exception as e:
            logger.error(f"Error during formula evaluation: {e}")
            result_df[self.new_column_name] = np.nan

        return result_df
