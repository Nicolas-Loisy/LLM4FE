import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from feature_engineering.transformation_factory import TransformationFactory


df = pd.DataFrame({
    'price': [10, 20, 30],
    'tax': [2, 3, 5]
})

config = {
    'finalCol': 'total_price',
    'colToProcess': ['price', 'tax'],
    'providerTransform': 'add', 
    'param': None 
}

factory = TransformationFactory()
factory.create_transformation(config)
df_transformed = factory.apply_transformations(df)

print(df_transformed)
