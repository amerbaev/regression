import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

SAMPLE_SIZE = 300
TEST_SIZE = 1

if __name__ == '__main__':
    sunspot_df = pd.read_csv('zuerich-monthly-sunspot-numbers-.csv')
    sunspot_df.drop(sunspot_df.index[[-1]], inplace=True)
    values = sunspot_df['Zuerich monthly sunspot numbers 1749-1983']
    plt.plot(values.index, values.values)

    for i in range(len(values) - SAMPLE_SIZE):
        sample = values[i:i+SAMPLE_SIZE]
        x_train, x_test = np.array(sample[:SAMPLE_SIZE-TEST_SIZE].index).reshape(-1, 1), np.array(sample[SAMPLE_SIZE-TEST_SIZE:].index).reshape(-1, 1)
        y_train, y_test = sample[:-TEST_SIZE].values, sample[-TEST_SIZE:].values

        regr = LinearRegression()
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        # plt.scatter(x_train, y_train)
        # plt.scatter(x_test, y_test)
        plt.scatter(x_test, y_pred)

    plt.show()