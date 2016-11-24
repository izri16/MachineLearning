import pandas as pd
import numpy as np
import csv

from firstHourAll import *

data = []
hour_position = 1
author_position = 2
section_position = 3
day_position = 0

with open('immediate_data_mt10click.csv', newline='') as csvfile:
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
author_d = np.array(pd.get_dummies(
    preprocess_authors_sections(X[:, author_position], 50)))

# section dummies
section_d = np.array(pd.get_dummies(X[:, section_position]))

days = X[:, day_position]
# remove day from features, only used to divide to training and testing data
X = np.concatenate((X[:, 0:day_position], hours_d,
                    X[:, ref_position:author_position], author_d,
                    section_d), axis=1)

X = X.astype(float)
y = y.astype(np.float)

print(X.shape)

# get only clicks, unique users and acceleration for regression
# X = get_basic_features(X)

X = get_polynomial_features(X)
X_train, X_test, y_train, y_test = split_set(X, y, days)

y_predicted = predict_random_forest(X_train, y_train, X_test)
predicted = clear_predictions(y_predicted, y)

show_results(y_test, predicted)
plot_results(y_test, predicted)
