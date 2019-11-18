import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# converts pd.Series of dates into the number of seconds since the earliest time in the series
def date_to_seconds(date_time: pd.Series) -> pd.Series:
    date_time = pd.to_datetime(date_time)
    date_time = date_time.astype(np.int64)
    temp = date_time.sub(date_time.min()) // 10 ** 9

    return temp


# data_lists -> list containing lists of data
def lists_to_df(data_lists: list, col_names: list) -> pd.DataFrame:
    df = pd.DataFrame()
    for data_list, name in zip(data_lists, col_names):
        df[name] = data_list

    return df


def x_y_split(df: pd.DataFrame, y_column: str) -> (pd.Series, pd.Series):
    names = list(df.columns)
    names.remove(y_column)

    X = df[names]
    y = df[y_column]

    return X, y


class PolynomialRegression:
    def __init__(self, x: pd.Series, y: pd.Series, degree: int):
        self.polynomial_features = self.create_polynomial_features(x, y, degree)
        self.model = self.create_model(x, y)

    def create_polynomial_features(self, x: pd.Series, y: pd.Series, degree) -> PolynomialFeatures:
        x = x.values.reshape(-1, 1) if len(x.columns) == 1 else x.values
        y = y.values

        polynomial_features = PolynomialFeatures(degree=degree)
        x_polynomial_features = polynomial_features.fit_transform(x)

        polynomial_features.fit(x_polynomial_features, y)

        return polynomial_features

    def create_model(self, x: pd.Series, y: pd.Series) -> LinearRegression:
        x = x.values.reshape(-1, 1) if len(x.columns) == 1 else x.values
        y = y.values

        model = LinearRegression()
        model.fit(self.polynomial_features.fit_transform(x), y)

        return model

    # don't know if it works for lists of lists only have tried single list
    def predict(self, x_to_predict: list):
        predictions = []
        for x_predict in x_to_predict:
            x_predict = np.array(x_predict)
            x_predict = x_predict.reshape(1, -1)

            predictions.append(self.model.predict(self.polynomial_features.fit_transform(x_predict))[0])

        return predictions

if __name__ == "__main__":
    x = [1, 2, 3]
    #x2 = [1, 2, 3]
    y = [3, 2, 1]



    names = ['x', 'y']

    df = lists_to_df([x, y], names)
    x, y = x_y_split(df, 'y')


    n = PolynomialRegression(x, y, degree=4)

    pred = [2, 2, 1]

    print(n.predict(pred))

'''
        y = pd.to_numeric(y, errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna()
'''
