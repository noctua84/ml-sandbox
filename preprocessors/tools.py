import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def import_csv_dataset(path: str):
    """Import a dataset from a CSV file"""
    def delimiter_check():
        """Check the delimiter of the CSV file"""
        with open(path, 'r') as file:
            first_line = file.readline()
            if first_line.count(",") > first_line.count(";"):
                return ","
            else:
                return ";"

    dataset = pd.read_csv(path, delimiter=delimiter_check())

    return dataset, analyse_dataset(dataset)


def analyse_dataset(dataset: pd.DataFrame):
    """Analyse the dataset"""

    missing_count = dataset.isnull().sum()
    available_columns = dataset.columns
    cols_with_missing_data = dataset.columns[dataset.isnull().any()]

    return {
        "missing_count": missing_count,
        "available_columns": available_columns,
        "missing_data": cols_with_missing_data
    }


def missing_numerical_data(matrix: np.array):
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


def encoding_categorical_data(matrix: np.array):
    """Encode categorical data"""
    def is_categorical(cur_col):
        """Check if the column is categorical"""
        try:
            cur_col.astype(float)
            return False
        except ValueError:
            return True

    categorical_cols = [i for i in range(matrix.shape[1]) if is_categorical(matrix[:, i])]

    if categorical_cols:
        transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)],
                                        remainder='passthrough')
        matrix = transformer.fit_transform(matrix)

    return matrix


def encode_dependent_vector(vector: np.array):
    """Encode the dependent vector"""
    encoder = LabelEncoder()
    return encoder.fit_transform(vector)


def split_dataset(x, y, test_size=0.2, random_state=1):
    """Split the dataset into training and test sets"""
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def feature_scaling(x_test, x_train):
    """Feature scaling"""
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_test, x_train
