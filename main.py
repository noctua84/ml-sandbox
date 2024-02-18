from preprocessors import tools

if __name__ == "__main__":
    data_set = "data/Data.csv"
    raw_x, raw_y = tools.import_csv_dataset(data_set)

    refined_x = tools.missing_data(raw_x)
    print(refined_x)
