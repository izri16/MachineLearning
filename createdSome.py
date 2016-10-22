import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split

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

with open('dataCreated.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in reader:
        if (row):
            if (i != 0):
                data.append(row)
            i += 1

data = np.array(data)
y = data[:, len(data[0, :])-1] # last columns
X = data[:, 1:len(data[0, :])-1] # do not use day feature
ones = np.ones((len(X), 1)) # add column of ones
X = np.concatenate([ones, X], axis=1)

# day time dummies
hours = list(X[:, 1].astype(float))
hours = np.array(list(map(day_times, hours)))
hours_d = np.array(pd.get_dummies(hours))

# author dummies
author_d = np.array(pd.get_dummies(X[:, 2]))

# section dummies
section_d = np.array(pd.get_dummies(X[:, 3]))

aaa = np.vander(X[:, 1].astype(float), 3)
aaa = aaa[:, 0:2]
print('a', aaa)

X = np.concatenate((X[:, 0:1],aaa , hours_d, author_d, section_d), axis=1)
X = X.astype(np.float)
y = y.astype(np.float)
print('X shape', X.shape, 'y shape', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
        random_state=42)

lr = lm.LinearRegression()
lr.fit(X_train, y_train)

y_predicted = lr.predict(X_test)

# print('Coefficients', lr.coef_)

res = abs(y_test - y_predicted)
error_m = np.mean(res)
error_r = np.sqrt(np.mean(res**2))
print('Mean error', error_m)
print('Root squared error', error_r)

# some values are predicted weird negative results
predicted = []
for i in list(y_predicted):
    if i > 0:
        predicted.append(i)
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
plt.title('Linear regression (Created data without author information)')
plt.xlabel('Page view id')
plt.ylabel('Total clicks')
plt.show()

