import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.feature_engineering.transformation_factory import TransformationFactory

df = pd.DataFrame({
    'price': [10, 20, 30],
    'tax': [2, 3, 5]
})

configs = [
        {
            'transformation_type': 'add',
            'description': 'Additionne les colonnes a et b',
            'category': 'math',
            'new_column_name': 'a_plus_b',
            'source_columns': ['price', 'tax'],
            'transformation_params': None
        },
        {
            'transformation_type': 'multiply',
            'description': 'Multiplie les colonnes a et b',
            'category': 'math',
            'new_column_name': 'a_times_b',
            'source_columns': ['price', 'tax'],
            'transformation_params': None
        }
    ]

factory = TransformationFactory()

for config in configs:
        factory.create_transformation(config)

df_transformed = factory.apply_transformations(df)

print("\nDataFrame transformé :")
print(df_transformed)