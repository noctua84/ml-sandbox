from ml_models import regression
from preprocessors import tools, plotter

if __name__ == "__main__":
    file = "data/Salary_Data.csv"
    df, inspection = tools.import_csv_dataset(file)

    matrix = df.iloc[:, :-1].values
    dependent_variable = df.iloc[:, -1].values

    matrix_train, matrix_test, vector_train, vector_test = tools.split_dataset(
        matrix, dependent_variable, test_size=1 / 3, random_state=0
    )

    test_pred, train_pred = regression.simple_linear_regression(matrix_train, vector_train, matrix_test)

    train_plt = plotter.scatter_plot(matrix_train, vector_train, train_pred, {
        "sc_color": "orange",
        "plt_color": "green",
        "title": "Salary vs Experience (Training set)",
        "x_label": "Years of Experience",
        "y_label": "Salary"
    })

    test_plt = plotter.scatter_plot(matrix_test, vector_test, test_pred, {
        "sc_color": "red",
        "plt_color": "blue",
        "title": "Salary vs Experience (Test set)",
        "x_label": "Years of Experience",
        "y_label": "Salary"
    })
