import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# from sklearn.model_selection import train_test_split

data = []


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
        if (int(value) in [31, 0, 1, 2, 3, 4]):
            x_train.append(xl[index])
            y_train.append(yl[index])
        else:
            x_test.append(xl[index])
            y_test.append(yl[index])
    return (np.array(x_train), np.array(x_test), np.array(y_train),
            np.array(y_test))

with open('dataCreated.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in reader:
        if (row):
            if (i != 0):
                data.append(row)
            i += 1

data = np.array(data)
y = data[:, len(data[0, :]) - 1]  # last columns
X = data[:, 0:len(data[0, :]) - 1]  # do not use day feature
ones = np.ones((len(X), 1))  # add column of ones
X = np.concatenate([ones, X], axis=1)

# day time dummies
hours = list(X[:, 1].astype(float))
hours = np.array(list(map(day_times, hours)))
hours_d = np.array(pd.get_dummies(hours))

# author dummies
author_d = np.array(pd.get_dummies(X[:, 2]))

# section dummies
section_d = np.array(pd.get_dummies(X[:, 3]))

days = X[:, 1]
print('ddd', author_d[0, :])
X = np.concatenate((X[:, 0:1], hours_d, author_d,
                    section_d), axis=1)
X = X.astype(np.float)
y = y.astype(np.float)
print('X shape', X.shape, 'y shape', y.shape)

X_train, X_test, y_train, y_test = split_set(X, y, days)

lr = lm.RidgeCV()
lr.fit(X_train, y_train)

y_predicted = lr.predict(X_test)

# print('Coefficients', lr.coef_)

# some values are predicted weird negative results
predicted = []
max_value = max(y)
for i in list(y_predicted):
    if i > 0 and i < max_value * 2:
        predicted.append(i)
    elif (i >= max_value * 2):
        predicted.append(max_value * 2)
    else:
        predicted.append(0)

predicted = np.array(predicted)
res = abs(y_test - predicted)
error_m = np.mean(res)
error_r = np.sqrt(np.mean(res**2))
print('Mean error (remove negatives)', error_m)
print('Root squared error (remove negatives)', error_r)

# Use when article is created
plt.figure(1)
plt.plot(y_test, 'ok', label='Test values')
plt.plot(predicted, 'or', label='Predicted values')
plt.legend(loc=1)
plt.title('Linear regression (Created data)')
plt.xlabel('Page view id')
plt.ylabel('Total clicks')
plt.show()
