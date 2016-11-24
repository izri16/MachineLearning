import sklearn.linear_model as lm
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.ensemble as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import metrics as mt

data = []
hour_position = 4
author_position = 9
section_position = 10
day_position = 3
ref_position = 5
min_predicted_value = 3


def day_times(hour):
    if (hour >= 6 and hour <= 11):
        return 1
    elif (hour >= 12 and hour <= 17):
        return 2
    elif (hour >= 18 and hour <= 22):
        return 3
    else:
        return 0


def split_set(x, y, days):
    xl = list(x)
    yl = list(y)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for index, value in enumerate(days):
        if (int(value) < 15):
            x_train.append(xl[index])
            y_train.append(yl[index])
        else:
            x_test.append(xl[index])
            y_test.append(yl[index])
    return (np.array(x_train), np.array(x_test), np.array(y_train),
            np.array(y_test))


def clear_predictions(y_predicted):
    '''
    some values can be predicted weird negative results
    or weird positive results
    '''
    predicted = []
    max_value = max(y)
    for i in list(y_predicted):
        if i > 0 and i < max_value * 2:
            predicted.append(i)
        elif (i >= max_value * 2):
            predicted.append(max_value * 2)
        else:
            predicted.append(min_predicted_value)
    return np.array(predicted)


def show_results(y_test, predicted):
    print('MAE', mt.mean_absolute_error(y_test, predicted))
    print('MSE', mt.mean_squared_error(y_test, predicted))
    rmspe = np.sqrt(
        np.mean((np.square(np.divide(y_test - predicted, y_test)))))
    print('RMSPE', rmspe)


def plot_results(y_test, predicted):
    plt.figure(1)
    plt.plot(y_test, 'ok', label='Test values')
    plt.plot(predicted, 'or', label='Predicted values')
    plt.legend(loc=2)
    plt.title('Linear regression (Created data)')
    plt.xlabel('Page view id')
    plt.ylabel('Total clicks')
    # plt.xlim([0, 500])
    plt.show()


def predict_regression(x_train, y_train, x_test):
    lr = lm.LinearRegression()
    lr.fit(x_train, y_train)
    return lr.predict(x_test)


def predict_neural(x_train, y_train, x_test):
    nm = nn.MLPRegressor()
    nm.fit(x_train, y_train)
    return nm.predict(x_test)


def predict_svm(x_train, y_train, x_test):
    svr = svm.SVR(C=10000, gamma=0.000000001)
    svr.fit(x_train, y_train)
    return svr.predict(x_test)


def predict_random_forest(x_train, y_train, x_test):
    rf = dt.RandomForestRegressor()
    rf.fit(x_train, y_train)
    return rf.predict(x_test)


def predict_boosted_tree(x_train, y_test, x_test):
    bt = dt.GradientBoostingRegressor()
    bt.fit(x_train, y_train)
    return bt.predict(x_test)

with open('first_hour_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in reader:
        if (row):
            if (i != 0):
                data.append(row)
            i += 1

data = np.array(data)
y = data[:, -1]
X = data[:, 0:-1]

# day time dummies
hours = list(X[:, hour_position].astype(float))
hours_labeled = np.array(list(map(day_times, hours)))
hours_d = np.array(pd.get_dummies(hours_labeled))

# author dummies
author_d = np.array(pd.get_dummies(X[:, author_position]))

# section dummies
section_d = np.array(pd.get_dummies(X[:, section_position]))

# polynomial features
# clicks_p = np.vander(X[:, 1].astype(float), 3)
# users_p = np.vander(X[:, 2].astype(float), 3)
# clicks_p = clicks_p[:, 0:2]
# users_p = users_p[:, 0:2]
# c_u = np.multiply(clicks_p[:, 0:1], users_p[:, 0:1])

days = X[:, day_position]
# remove day from features, only used to divide to training and testing data
X = np.concatenate((X[:, 0:day_position], hours_d,
                    X[:, ref_position:author_position], author_d,
                    section_d), axis=1)

X = X.astype(float)
y = y.astype(np.float)

X_train, X_test, y_train, y_test = split_set(X, y, days)

y_predicted = predict_regression(X_train, y_train, X_test)
predicted = clear_predictions(y_predicted)

show_results(y_test, predicted)
plot_results(y_test, predicted)
