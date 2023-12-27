import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def read_data(path:str) -> pd.DataFrame:
    """
    Read in the dataset from a selected directory.
    Adjust dataframe to specific format.
    :param path: String containing the path name.
    :return: Adjusted dataset pulled from directory path.
    """
    # Read in data from path
    df = pd.read_csv(path)
    df = df.drop("image_id", axis=1)

    # Shuffle the data
    df = df.sample(frac=1)
    df = pd.get_dummies(df)

    return df

def split_data(df:pd.DataFrame,ratio:float,target:str) -> (pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame):
    """
    Split a dataframe by target and predictor variables.
    Decide which data will be test data and which data will be training data.
    :param df: Dataframe to be split.
    :param ratio: Ratio (float) of split between training and testing.
    :param target: String of target column name.
    :return: 4 dataframes - the train data for X and y, the test data for X and y.
    """
    X = df[df.columns[~df.columns.isin([target])]]
    y = df[[target]]
    train_X,test_X, train_y,test_y = train_test_split(X,y,test_size=ratio)
    return train_X,test_X, train_y,test_y

def train_data_XGB(train_X:pd.DataFrame,train_y:pd.DataFrame):
    """
    Build the model using XGBoost.
    Print out the accuracy of the model.
    Return the model.
    :param train_X: Train data for X.
    :param train_y: Train data for y.
    :return: Prediction model.
    """
    model = XGBClassifier()
    model.fit(train_X, train_y)
    return model

def possess(model,test_X:pd.DataFrame,test_y:pd.DataFrame):
    """
    Take in the fitted model and use test data to output the models accuracy.
    :param model: Already built model.
    :param test_X: train data for X.
    :param test_y: train data for y.
    :return:
    """
    y_pred = model.predict(test_X)
    predictions = [round(value) for value in y_pred]
    # Compare for accuracy
    accuracy = accuracy_score(test_y, predictions)
    return accuracy

def predict(model, df:pd.DataFrame):
    """
    Take in the model and a dataset of value to use for prediction.
    Use the model on the dataset and return a binary vector of predictions.
    :param model: Already built model.
    :param df: Dataframe specific to the models features.
    :return: Binary vector of predictions.
    """
    # Fit the model to the dataset that will be used for prediction
    predictions = model.predict(df)
    predictions = [round(value) for value in predictions]
    # Return a binary vector of predictions
    return predictions

if __name__ == '__main__':

    # Read in data
    df = read_data("/Users/zachstrennen/Downloads/Subject #.csv")

    # Establish clean data frame to default to
    df_clean = df

    # Split and train the data
    train_X, test_X, train_y, test_y = split_data(df, 0.8, "beauty_classification")
    model = train_data_XGB(train_X, train_y)

    accuracy = possess(model, test_X, test_y)

    # Original Accuracy before features are limited
    print("Original Accuracy:")
    print(accuracy)


    """
    The following loop creates multiple models for each feature.
    The mean accuracy of each feature is stored in a list.
    """
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
    index_list=[0]

    # Create a list of all indices scoring above the threshold
    for i in range(0,len(mean_list)):
        if mean_list[i] > threshold:
            index_list.append(i+1)

    """
    The following loop builds fifty models using the selected features only.
    The output at the end of the loop prints the final average accuracy score to be compared to the original.
    Additionally, all features' indices are noted.
    """
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
    print("Final Accuracy:")
    print(mean_acc)

    print("Important Feature Indices:")
    print(index_list)

        