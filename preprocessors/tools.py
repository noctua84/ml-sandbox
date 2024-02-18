import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def import_csv_dataset(path: str):
    """Import a dataset from a CSV file"""
    dataset = pd.read_csv(path)
    feature_matrix = dataset.iloc[:, :-1].values
    dependent_vector = dataset.iloc[:, -1].values

    return feature_matrix, dependent_vector


def missing_data(matrix: np.array):
    """Replace missing data with the mean of the column"""
    def is_numerical(cur_col):
        """Check if the column is numerical"""
        try:
            cur_col.astype(float)
            return True
        except ValueError:
            return False

    numerical_cols = [i for i in range(matrix.shape[1]) if is_numerical(matrix[:, i])]

    if numerical_cols:
        for col in numerical_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(matrix[:, col:col+1])
            matrix[:, col:col+1] = imputer.transform(matrix[:, col:col+1])

    return matrix
