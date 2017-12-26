import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLE_SIZE = 100
TEST_SIZE = 1

if __name__ == '__main__':
    sunspot_df = pd.read_csv('zuerich-monthly-sunspot-numbers-.csv')
    sunspot_df.drop(sunspot_df.index[[-1]], inplace=True)
    Y = sunspot_df['count']
    # print(X, Y)
    sns.set()
    sunspot_df.plot(x=['year', 'month'], y='count')
    test = []
    pred = []

    for i in range(len(Y) - SAMPLE_SIZE):
        sample = sunspot_df[i:i+SAMPLE_SIZE]

        x_train, x_test = sample[['year', 'month']][:-TEST_SIZE], sample[['year', 'month']][-TEST_SIZE:]
        y_train, y_test = sample['count'][:-TEST_SIZE], sample['count'][-TEST_SIZE:]

        lin_regr = LinearRegression()
        lin_regr.fit(x_train, y_train)
        lin_y_pred = lin_regr.predict(x_test)
        test.append(y_test)
        pred.append(lin_y_pred)
        plt.plot(x_test, lin_y_pred, color='red')

    # print('Среднеквадратичное отклонение: ', mean_squared_error(test, pred))
    # print('Дисперсия:', r2_score(test, pred))
    plt.show()