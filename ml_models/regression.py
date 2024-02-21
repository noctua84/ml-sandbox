from sklearn.linear_model import LinearRegression


def simple_linear_regression(matrix_train, vector_train, matrix_test):
    """
    Simple linear regression
    The simple linear regression best fits for linear datasets
    """
    regressor = LinearRegression()
    regressor.fit(matrix_train, vector_train)
    vector_test_pred = regressor.predict(matrix_test)
    vector_train_pred = regressor.predict(matrix_train)

    return vector_test_pred, vector_train_pred
