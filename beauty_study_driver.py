import pandas as pd
from beauty_study_functions import read_data, split_data,\
    train_data_XGB, possess


if __name__ == '__main__':

    # Read in data
    df = read_data("data/CHOOSE SUBJECT FROM FILE")

    # Establish clean data frame to default to
    df_clean = df

    # Split and train the data
    train_X, test_X, train_y, test_y = split_data(df, 0.8, "beauty_classification")
    model = train_data_XGB(train_X, train_y)

    accuracy = possess(model, test_X, test_y)

    # Original Accuracy before features are limited
    print("Original Accuracy:")
    print(accuracy)

    # The following loop creates multiple models for each feature.
    # The mean accuracy of each feature is stored in a list.
    mean_list = []
    for i in range(1, 27):
        df = df_clean

        # Focus on a specific feature
        df = df.iloc[:, [0, i]]
        col_name = df.columns[1]
        df = pd.get_dummies(df)
        sum_acc = 0

        # Create multiple models for the feature and store the mean accuracy
        for j in range(0, 50):
            df = df.sample(frac=1)
            train_X, test_X, train_y, test_y = split_data(df, 0.8, "beauty_classification")
            model = train_data_XGB(train_X, train_y)
            accuracy = possess(model, test_X, test_y)
            sum_acc = sum_acc + accuracy

        mean_acc = sum_acc / 50
        mean_list.append(mean_acc)

    # Threshold to be tuned if it helps accuracy
    # Mean of accuracy means is baseline threshold here
    threshold = sum(mean_list)/len(mean_list)
    index_list = [0]

    # Create a list of all indices scoring above the threshold
    for i in range(0, len(mean_list)):
        if mean_list[i] > threshold:
            index_list.append(i+1)

    # The following loop builds fifty models using the selected features only.
    # The output at the end of the loop prints the final average accuracy
    # score to be compared to the original.
    # Additionally, all features' indices are noted.
    sum_acc = 0
    for i in range(0, 50):
        df = df_clean
        df = df.iloc[:, index_list]
        df = pd.get_dummies(df)
        train_X, test_X, train_y, test_y = split_data(df, 0.8, "beauty_classification")
        model = train_data_XGB(train_X, train_y)
        accuracy = possess(model, test_X, test_y)
        sum_acc = sum_acc + accuracy

    mean_acc = sum_acc / 50

    # Print Results
    print("Final Accuracy:")
    print(mean_acc)

    print("Important Feature Indices:")
    print(index_list)
