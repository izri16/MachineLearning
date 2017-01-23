import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.ensemble as dt
import pandas as pd
import numpy as np
import sklearn.linear_model as rd
import matplotlib.pyplot as plt
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


def print_results(y_test, pred_test, y_train, pred_train):
    d = {
        'Train:': [y_train, pred_train],
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


def predict_regression(x_train, y_train, x_test):
    lr = rd.Ridge()

    lr.fit(x_train, y_train)
    pred_test = lr.predict(x_test)
    pred_train = lr.predict(x_train)
    return (pred_train, pred_test)


def predict_neural(x_train, y_train, x_test):
    nm = nn.MLPRegressor(hidden_layer_sizes=(100, 100))

    nm.fit(x_train, y_train)
    pred_test = nm.predict(x_test)
    pred_train = nm.predict(x_train)
    return (pred_train, pred_test)


def predict_svm(x_train, y_train, x_test):
    svr = svm.SVR(C=300, kernel='linear')

    svr.fit(x_train, y_train)
    pred_test = svr.predict(x_test)
    pred_train = svr.predict(x_train)
    return (pred_train, pred_test)


def predict_random_forest(x_train, y_train, x_test):
    rf = dt.RandomForestRegressor(min_samples_leaf=5, n_estimators=150,
                                  max_depth=15)

    rf.fit(x_train, y_train)
    pred_test = rf.predict(x_test)
    pred_train = rf.predict(x_train)
    return (pred_train, pred_test)


def predict_boosted_tree(x_train, y_train, x_test):
    bt = dt.GradientBoostingRegressor(min_samples_leaf=5, max_depth=11,
                                      n_estimators=300)

    bt.fit(x_train, y_train)
    pred_test = bt.predict(x_test)
    pred_train = bt.predict(x_train)
    return (pred_train, pred_test)


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
    data_sept = pd.read_csv('./data_september_authors.csv')
    data_oct = pd.read_csv('./data_october_authors.csv')
    data_nov = pd.read_csv('./data_november_authors.csv')

    data = pd.concat([data_sept, data_oct, data_nov])

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

    x_train = x.iloc[0:data_sept.shape[0], :]
    x_test = x.iloc[data_sept.shape[0]:data.shape[0], :]

    y_train = y.iloc[0:data_sept.shape[0]].values
    y_test = y.iloc[data_sept.shape[0]:data.shape[0]].values

    # SAVE UNSCALLED DATA TO FILE
    # np.savetxt("x_train.csv", x_train, delimiter=",")
    # np.savetxt("y_train.csv", y_train, delimiter=",")
    # np.savetxt("x_test.csv", x_test, delimiter=",")

    # SCALE DATA TO HAVE ZERO MEAN AND UNIT VARIANCE
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # SAVE SCALLED DATA TO FILE
    # np.savetxt("x_train_scalled.csv", x_train, delimiter=",")
    # np.savetxt("y_train_scalled.csv", y_train, delimiter=",")
    # np.savetxt("x_test_scalled.csv", x_test, delimiter=",")

    # RANDOM FOREST
    pred_train, pred_test = predict_random_forest(x_train, y_train, x_test)

    # SVR
    # pred_train, pred_test = predict_svm(x_train, y_train, x_test)

    # BOOSTED TREE
    # pred_train, pred_test = predict_boosted_tree(x_train, y_train, x_test)

    # ANN
    # pred_train, pred_test = predict_neural(x_train, y_train, x_test)

    # REGRESSION
    pred_train, pred_test = predict_regression(x_train, y_train, x_test)

    # ROUND ALL VALUES TO INTEGERS
    pred_test = clear_predictions(np.around(pred_test, 0).astype(int), y_test)
    pred_train = clear_predictions(
        np.around(pred_train, 0).astype(int), y_train)

    print('Min value in test set', min(y_test))
    print('Max value in test set', max(y_test))
    print('Min predicted value', min(pred_test))
    print('Max predicted value', max(pred_test))

    print_results(y_test, pred_test, y_train, pred_train)
    plot_results(y_test, pred_test, y_train, pred_train)
