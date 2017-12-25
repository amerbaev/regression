import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLE_SIZE = 500
TEST_SIZE = 1

if __name__ == '__main__':
    sunspot_df = pd.read_csv('zuerich-monthly-sunspot-numbers-.csv')
    sunspot_df.drop(sunspot_df.index[[-1]], inplace=True)
    values = sunspot_df['Zuerich monthly sunspot numbers 1749-1983']
    sns.set()
    plt.plot(values.index, values.values)

    for i in range(len(values) - SAMPLE_SIZE):
        sample = values[i:i+SAMPLE_SIZE]
        x_train, x_test = np.array(sample[:SAMPLE_SIZE-TEST_SIZE].index).reshape(-1, 1), np.array(sample[SAMPLE_SIZE-TEST_SIZE:].index).reshape(-1, 1)
        y_train, y_test = sample[:-TEST_SIZE].values, sample[-TEST_SIZE:].values

        lin_regr = LinearRegression()
        lin_regr.fit(x_train, y_train)
        lin_y_pred = lin_regr.predict(x_test)
        plt.scatter(x_test, lin_y_pred, color='red')

    plt.show()