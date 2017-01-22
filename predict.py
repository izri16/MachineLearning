import sklearn.linear_model as lm
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.ensemble as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as mt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

TRAIN_END = 2000
VAL_END = 3000


def get_day_times(hour):
    if (hour >= 6 and hour <= 11):
        return 'morning'
    elif (hour >= 12 and hour <= 17):
        return 'afternoon'
    elif (hour >= 18 and hour <= 22):
        return 'evening'
    else:
        return 'night'


def clear_predictions(y_predicted, y):
    '''
    some values can be predicted weird negative results
    or weird positive results
    '''
    min_predicted_value = 3
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
    print('MAE', round(mt.mean_absolute_error(y_test, predicted), 3))
    rmspe = np.sqrt(
        np.mean((np.square(np.divide(y_test - predicted, y_test)))))
    print('RMSPE', round(rmspe, 5) * 100)


def plot_results(y_test, predicted):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(y_test, 'ok', label='Test values')
    plt.plot(predicted, 'or', label='Predicted values')
    plt.legend(loc=1)
    plt.title('Results')
    plt.xlabel('Page view id')
    plt.ylabel('Total clicks')
    plt.ylim([0, 500])

    plt.subplot(212)
    plt.plot(y_test, 'ok', label='Test values')
    plt.plot(predicted, 'or', label='Predicted values')
    plt.legend(loc=1)
    plt.xlabel('Page view id')
    plt.ylabel('Total clicks')
    plt.show()


def get_most_important_features(x, y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(x, y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(25, 15))
    plt.title('Feature importances')
    plt.bar(range(x_train.shape[1]), importances[indices],
            color='r', yerr=std[indices], align='center')
    plt.xticks(range(x_train.shape[1]), np.array(x.columns)[indices],
               rotation='vertical')
    plt.xlim([-1, x_train.shape[1]])
    plt.savefig('feature_importances.png')


def predict_regression(x_train, y_train, x_test):
    lr = lm.LinearRegression()
    lr.fit(x_train, y_train)
    return lr.predict(x_test)


def predict_neural(x_train, y_train, x_test):
    nm = nn.MLPRegressor(
        alpha=0.0001, hidden_layer_sizes=(100, 100),
        max_iter=1000, activation='relu')
    nm.fit(x_train, y_train)
    return nm.predict(x_test)


def predict_svm(x_train, y_train, x_test):
    # RBF kernel
    svr = svm.SVR(C=5000, gamma=0.0075)
    svr.fit(x_train, y_train)
    return svr.predict(x_test)


def predict_random_forest(x_train, y_train, x_test):
    rf = dt.RandomForestRegressor(
        n_estimators=300, max_depth=30)
    rf.fit(x_train, y_train)
    return rf.predict(x_test)


def predict_boosted_tree(x_train, y_train, x_test):
    bt = dt.GradientBoostingRegressor(n_estimators=150, max_depth=10)
    bt.fit(x_train, y_train)
    return bt.predict(x_test)


def filter_sections(section):
    if (section == 'sport'):
        return 'sport'
    elif (section == 'news'):
        return 'news'
    elif (section == 'whats-on'):
        return 'whats-on'
    else:
        return 'other'

if __name__ == "__main__":
    scaler = StandardScaler()
    data = pd.read_csv('./data_september.csv')

    # day time dummies
    day_times = list(map(get_day_times, data['hour'].tolist()))
    data['hour'] = pd.Series(day_times).values
    data = pd.get_dummies(data, columns=['hour'], drop_first=True)

    # section dummies
    filtered_sections = list(
        map(filter_sections, data['contentSection'].tolist()))
    data['contentSection'] = pd.Series(filtered_sections).values
    data = pd.get_dummies(data, columns=['contentSection'], drop_first=True)

    data = data.fillna(0)  # fill empty values with zeros
    data = data.sort(['contentCreatedFixed'], ascending=[1])  # sort by date

    # Drop data that should not be used
    data = data.drop(['day', 'contentAuthor', 'contentCreatedFixed',
                      'contentSection_sport', 'contentSection_other',
                      'contentSection_whats-on',
                      'ref', 'desktop', 'mobile', 'tablet'], axis=1)

    data = data.drop(['hour_morning', 'hour_night', 'hour_evening'], axis=1)
    data = data.drop(['ref_internal', 'ref_direct',
                      'ref_external', 'ref_social'], axis=1)
    data = data.drop(['country_gb', 'country_us'], axis=1)
    data = data.drop(['incRatio'], axis=1)

    y = data['totalClicks']
    x = data.drop(['totalClicks'], axis=1)

    print(x.columns)

    x.to_csv('x.csv')
    y.to_csv('y.csv')

    x_train = x.iloc[0:TRAIN_END, :]
    x_val = x.iloc[TRAIN_END:VAL_END, :]

    # COMMENT OUT TO CORRECTLY USE TEST SET
    # x_test = data.iloc[VAL_END:data.shape[0], :]
    x_test = x.iloc[TRAIN_END:data.shape[0], :]

    y_train = y.iloc[0:TRAIN_END].values
    y_cv = y.iloc[TRAIN_END:VAL_END].values
    y_test = y.iloc[TRAIN_END:data.shape[0]].values

    # save numpy data to file
    np.savetxt("x_train.csv", x_train, delimiter=",")
    np.savetxt("y_train.csv", y_train, delimiter=",")
    np.savetxt("x_test.csv", x_test, delimiter=",")

    print(x_train.shape)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    print(x_train.shape)
    x_test = scaler.transform(x_test)
    print(x_test.shape)

    np.savetxt("x_train_sc.csv", x_train, delimiter=",")
    np.savetxt("y_train_sc.csv", y_train, delimiter=",")
    np.savetxt("x_test_sc.csv", x_test, delimiter=",")

    # GET MOST IMPORTANT FEATURES BASED ON RANDOM FOREST
    # get_most_important_features(x, y)

    # predicted = predict_random_forest(x_train, y_train, x_test)
    # predicted = predict_svm(x_train, y_train, x_test)
    # predicted = predict_boosted_tree(x_train, y_train, x_test)
    predicted = predict_neural(x_train, y_train, x_test)
    # predicted = predict_regression(x_train, y_train, x_test)

    predicted = clear_predictions(predicted, y_test)

    show_results(y_test, predicted)
    plot_results(y_test, predicted)
