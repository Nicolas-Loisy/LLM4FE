""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

loan_dts = pd.read_csv("data/datasets/loan_approval_dataset.csv")


# Suppression des espaces dans le noms des colonnes
loan_dts.columns = loan_dts.columns.str.strip()


# Vérification des valuers nulles
def missing_values(data):
    missing_values = data.isna().sum().sort_values(ascending=False)
    n_missing_values = missing_values[missing_values > 0]
    p_missing_values = (n_missing_values/data.shape[0])*100
    missing_data = pd.concat([n_missing_values, p_missing_values], axis=1, keys=[
                             'Count', 'Percentage'])
    return missing_data


print(missing_values(loan_dts))


# Vérification et conversion des type de données non numériques
print(loan_dts.info())
non_numeric_cols = loan_dts.select_dtypes(
    include=['object', 'category']).columns
encoder = OrdinalEncoder()
loan_dts[non_numeric_cols] = encoder.fit_transform(loan_dts[non_numeric_cols])
print(loan_dts.info())

answer = loan_dts["loan_status"]

loan_dts.drop(["loan_id", "loan_status"], axis=1, inplace=True)


train_data, val_data, train_answer, val_answer = train_test_split(
    loan_dts, answer, shuffle=True, random_state=42, test_size=0.20)
 """
from .prep import DatasetPrep
