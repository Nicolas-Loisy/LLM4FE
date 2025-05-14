import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


class DatasetPrep:

    def __init__(self, dataset, target_col):
        self._dataset = dataset
        self._target_col = target_col

    def get_init_dataset(self):
        data = pd.read_csv(f"data/datasets/{self._dataset}.csv")
        data.columns = data.columns.str.strip()
        return data

    def get_target_col(self):
        answer = self._dataset[f"{self._target_col}"]
        return answer

    def get_missing_values_free_dataset(data):
        missing_values = data.isna().sum().sort_values(ascending=False)
        n_missing_values = missing_values[missing_values > 0]
        if not n_missing_values.empty:
            data.fillna(0.0, inplace=True)
        return data

    def del_duplicate(self):
        if np.sum(self._dataset.duplicated()) >= 1:
            return self._dataset.drop_duplicates()
        else:
            return self._dataset

    def encoding_label_col(self):
        non_numeric_cols = self._dataset.select_dtypes(
            include=['object', 'category']).columns
        encoder = OrdinalEncoder()
        self._dataset[non_numeric_cols] = encoder.fit_transform(
            self._dataset[non_numeric_cols])
        print(self._dataset.info())
        return self._dataset
        # drop manuellement les col non nécessaire

    def cross_validation(self, size: float):
        models = {'gbc': {'model': GradientBoostingClassifier(), 'name': 'GradientBoostingClassifier'},
                  'xgb': {'model': XGBClassifier(), 'name': 'XGBClassifer'},
                  'rf': {'model': RandomForestClassifier(n_jobs=-1), 'name': 'RandomForestClassifier'},
                  'tree': {'model': DecisionTreeClassifier(), 'name': 'DecisionTreeClassifier'},
                  'knn': {'model': KNeighborsClassifier(), 'name': 'KNeighborsClassifier'},
                  'lr': {'model': LogisticRegression(max_iter=1000), 'name': 'LogisticRegression'},
                  'ss': {'model': SGDClassifier(), 'name': 'SGDClassifier'}
                  }
        best_model = None
        best_mean = -float('inf')
        best_std = float('inf')
        y = self._target_col
        for model in models:
            scores = cross_val_score(
                models[model]['model'], self._dataset, self._target_col, cv=4, scoring='accuracy')
            models[model]['score'] = scores
            mean_score = models[model]['score'].mean()
            std_score = models[model]['score'].std()**2
            print(f"{models[model]['name']} : {mean_score} (+/- {std_score})")
            if mean_score > best_mean or (mean_score == best_mean and std_score < best_std):
                best_model = models[model]
                best_mean = mean_score
                best_std = std_score
        print(
            f"Best model: {best_model['name']} with mean score: {best_mean} and std: {best_std}")
        tune = best_model['model']
        train_data, val_data, train_answer, val_answer = train_test_split(
            self._dataset, self._target_col, shuffle=True, random_state=42, test_size=0.20)
        tune.fit(train_data, train_answer)
        y_pred_val = tune.predict(val_data)
        val_score = accuracy_score(val_answer, y_pred_val)
        print("Accuracy sur la validation :", val_score)
        return val_score
