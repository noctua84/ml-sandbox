from preprocessors import tools

if __name__ == "__main__":
    # Prepare the dataset
    data_set = "data/Data.csv"
    dataset, analyse_data = tools.import_csv_dataset(data_set)
    feature_matrix = dataset.iloc[:, :-1].values
    dependent_vector = dataset.iloc[:, -1].values

    # Missing numerical data
    refined_feature_matrix = tools.missing_numerical_data(feature_matrix)

    # Encoding categorical data
    encoded_feature_matrix = tools.encoding_categorical_data(refined_feature_matrix)

    # Encoding dependent vector
    encoded_dependent_vector = tools.encode_dependent_vector(dependent_vector)

    # Splitting the dataset into the Training set and Test set
    train_matrix, test_matrix, train_vector, test_vector = tools.split_dataset(
        encoded_feature_matrix, encoded_dependent_vector)
    
    # Feature scaling
    train_matrix, test_matrix = tools.feature_scaling(test_matrix, train_matrix)
