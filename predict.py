import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.ensemble as dt
import pandas as pd
import numpy as np
import sklearn.linear_model as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn import metrics as mt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

TRAIN_END = 2000
VAL_END = 3500


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
    Some values can be predicted weird negative results
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


def print_results(y_test, pred_test, y_val, pred_val, y_train, pred_train):
    d = {
        'Train:': [y_train, pred_train],
        'Validation:': [y_val, pred_val],
        'Test:': [y_test, pred_test]
    }

    for set_type, sets in d.items():
        print()
        print(set_type)
        print('MAE', round(mt.mean_absolute_error(sets[0], sets[1]), 3))
        print('RMSE', round(np.sqrt(round(mt.mean_squared_error(sets[0],
                                                                sets[1]), 3)),
                            3))
        rmspe = np.sqrt(
            np.mean((np.square(np.divide(sets[0] - sets[1], sets[0])))))
        print('RMSPE', round(rmspe, 5) * 100)


def plot_results(y_test, pred_test, y_train, pred_train):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(y_test, 'ok', label='Test values')
    plt.plot(pred_test, 'or', label='Predicted values')
    plt.legend(loc=1)
    plt.title('Results')
    plt.xlabel('Article')
    plt.ylabel('Total clicks')

    plt.subplot(212)
    plt.plot(y_train, 'ok', label='Test values (Trainining set)')
    plt.plot(pred_train, 'or', label='Predicted values (Training set)')
    plt.legend(loc=1)
    plt.xlabel('Article')
    plt.ylabel('Total clicks')
    plt.show()


def get_most_important_features(x, y):
    forest = ExtraTreesClassifier(n_estimators=300,
                                  max_depth=10,
                                  random_state=0)
    forest.fit(x, y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(25, 15))
    plt.title('Feature importances')
    plt.bar(range(x.shape[1]), importances[indices],
            color='r', yerr=std[indices], align='center')
    plt.xticks(range(x.shape[1]), np.array(x.columns)[indices],
               rotation='vertical')
    plt.xlim([-1, x.shape[1]])
    plt.savefig('feature_importances.png')


def get_grid_search_data(x_train, y_train, x_val, y_val):
    # allows to use validation set instead of cross-validation
    # using GridSearchCV
    index_train = [-1] * len(x_train)
    index_validate = [0] * len(x_val)
    indexes = index_train + index_validate
    ps = PredefinedSplit(test_fold=indexes)

    x = np.append(x_train, x_val, axis=0)
    y = np.append(y_train, y_val, axis=0)
    score = 'r2'

    return (x, y, ps, score)


def predict_regression(x_train, y_train, x_val, y_val, x_test):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_val, y_val)

    lr = rd.Ridge()

    parameters = [
        {'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 9, 27, 50, 100]}
    ]

    clf = GridSearchCV(lr, parameters, scoring=score, cv=ps)
    clf.fit(x, y)
    pred_test = clf.predict(x_test)
    pred_val = clf.predict(x_val)
    pred_train = clf.predict(x_train)

    print('Best params found for Ridge Regression', clf.best_params_)
    return (pred_train, pred_val, pred_test)


def predict_neural(x_train, y_train, x_val, y_val, x_test):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_val, y_val)

    nm = nn.MLPRegressor()

    parameters = [
        {'alpha': [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001,
                   0.003, 0.01, 0.03],
         'max_iter': [400],
         'activation': ['relu'],
         'hidden_layer_sizes': [[30], [50], [75], [100], [30, 30],
                                [50, 50], [75, 75], [100, 100]],
         }
    ]

    clf = GridSearchCV(nm, parameters, scoring=score, cv=ps)
    clf.fit(x, y)
    pred_test = clf.predict(x_test)
    pred_val = clf.predict(x_val)
    pred_train = clf.predict(x_train)

    print('Best params found for Neural Network', clf.best_params_)
    return (pred_train, pred_val, pred_test)


def predict_svm(x_train, y_train, x_val, y_val, x_test):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_val, y_val)

    svr = svm.SVR()
    c = [1, 10, 30, 100, 300, 1000, 3000, 5000]

    # try to combine
    parameters = [
        {'C': c,
         'kernel': ['rbf'],
         'gamma': [0.0001, 0.003, 0.001, 0.003,
                   0.01, 0.03, 0.1, 0.3, 1, 3]},
        {'C': c,
         'kernel': ['linear']
         }
    ]

    clf = GridSearchCV(svr, parameters, scoring=score, cv=ps)
    clf.fit(x, y)
    pred_test = clf.predict(x_test)
    pred_val = clf.predict(x_val)
    pred_train = clf.predict(x_train)

    print('Best params found for SVM', clf.best_params_)
    return (pred_train, pred_val, pred_test)


def predict_random_forest(x_train, y_train, x_val, y_val, x_test):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_val, y_val)

    rf = dt.RandomForestRegressor()

    parameters = [
        {'min_samples_leaf': [5, 6, 7, 10],
         'n_estimators': [150, 300],
         'max_depth': [5, 7, 9, 11, 13, 15]}
    ]

    clf = GridSearchCV(rf, parameters, scoring=score, cv=ps)
    clf.fit(x, y)
    pred_test = clf.predict(x_test)
    pred_val = clf.predict(x_val)
    pred_train = clf.predict(x_train)

    print('Best params found for Random Forest', clf.best_params_)
    return (pred_train, pred_val, pred_test)


def predict_boosted_tree(x_train, y_train, x_val, y_val, x_test):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_val, y_val)

    bt = dt.GradientBoostingRegressor()

    parameters = [
        {'min_samples_leaf': [3, 4, 5, 6, 7],
         'n_estimators': [150, 300],
         'max_depth': [5, 7, 9, 11, 13, 15]}
    ]

    clf = GridSearchCV(bt, parameters, scoring=score, cv=ps)
    clf.fit(x, y)
    pred_test = clf.predict(x_test)
    pred_val = clf.predict(x_val)
    pred_train = clf.predict(x_train)

    print('Best params found for Boosted Tree', clf.best_params_)
    return (pred_train, pred_val, pred_test)


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

    # DAY TIME DUMMIES
    day_times = list(map(get_day_times, data['hour'].tolist()))
    data['hour'] = pd.Series(day_times).values
    data = pd.get_dummies(data, columns=['hour'], drop_first=True)

    # SECTION DUMMIES
    filtered_sections = list(
        map(filter_sections, data['contentSection'].tolist()))
    data['contentSection'] = pd.Series(filtered_sections).values
    data = pd.get_dummies(data, columns=['contentSection'], drop_first=True)

    data = data.fillna(0)  # fill empty values with zeros
    data = data.sort_values(by='contentCreatedFixed',
                            ascending=True)  # sort by date

    # DROP FEATURES THAT SHOULD NOT BE USED
    data = data.drop(['day', 'contentAuthor', 'contentCreatedFixed'], axis=1)

    # data = data.drop(['ref', 'desktop', 'mobile', 'tablet'], axis=1)

    data = data.drop(['contentSection_other',
                      'contentSection_sport',
                      'contentSection_whats-on'], axis=1)

    data = data.drop(['hour_night', 'hour_morning', 'hour_evening'], axis=1)

    data = data.drop(['ref_social'], axis=1)
    '''
    data = data.drop(['ref_internal', 'ref_direct',
                      'ref_external', 'ref_social'], axis=1)
    '''
    # data = data.drop(['country_gb', 'country_us'], axis=1)
    # data = data.drop(['incRatio'], axis=1)

    y = data['totalClicks']
    x = data.drop(['totalClicks'], axis=1)
    # x = data[['clicksFirstHour', 'uniqueUsers', 'incRatio']]

    # GET MOST IMPORTANT FEATURES BASED ON RANDOM FOREST
    # get_most_important_features(x, y)

    x_train = x.iloc[0:TRAIN_END, :]
    x_val = x.iloc[TRAIN_END:VAL_END, :]
    x_test = x.iloc[VAL_END:data.shape[0], :]

    y_train = y.iloc[0:TRAIN_END].values
    y_val = y.iloc[TRAIN_END:VAL_END].values
    y_test = y.iloc[VAL_END:data.shape[0]].values

    # SAVE UNSCALLED DATA TO FILE
    # np.savetxt("x_train.csv", x_train, delimiter=",")
    # np.savetxt("y_train.csv", y_train, delimiter=",")
    # np.savetxt("x_test.csv", x_test, delimiter=",")

    # SCALE DATA TO HAVE ZERO MEAN AND UNIT VARIANCE
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # SAVE SCALLED DATA TO FILE
    # np.savetxt("x_train_scalled.csv", x_train, delimiter=",")
    # np.savetxt("y_train_scalled.csv", y_train, delimiter=",")
    # np.savetxt("x_test_scalled.csv", x_test, delimiter=",")

    # RANDOM FOREST
    pred_train, pred_val, pred_test = predict_random_forest(
        x_train, y_train, x_val, y_val, x_test)

    # SVR
    '''
    pred_train, pred_val, pred_test = predict_svm(
        x_train, y_train, x_val, y_val, x_test)
    '''

    # BOOSTED TREE
    '''
    pred_train, pred_val, pred_test = predict_boosted_tree(
        x_train, y_train, x_val, y_val, x_test)
    '''

    # ANN
    '''
    pred_train, pred_val, pred_test = predict_neural(
        x_train, y_train, x_val, y_val, x_test)
    '''

    # REGRESSION
    '''
    pred_train, pred_val, pred_test = predict_regression(
        x_train, y_train, x_val, y_val, x_test)
    '''

    # ROUND ALL VALUES TO INTEGERS
    pred_test = clear_predictions(np.around(pred_test, 0).astype(int), y_test)
    pred_train = clear_predictions(
        np.around(pred_train, 0).astype(int), y_train)

    print('Min value in test set', min(y_test))
    print('Max value in test set', max(y_test))
    print('Min predicted value', min(pred_test))
    print('Max predicted value', max(pred_test))

    print_results(y_test, pred_test, y_val, pred_val, y_train, pred_train)
    plot_results(y_test, pred_test, y_train, pred_train)
