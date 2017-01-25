import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.ensemble as dt
import pandas as pd
import numpy as np
import sklearn.linear_model as rd
import matplotlib.pyplot as plt
import datetime
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


def get_most_important_articles(target, y_test, p_choose,
                                complete_test_data, x_train, hour=False):
    '''
    Return dictionary of 'p_choose' most important articles for every day.
    Days are keys in the dict and value is array of indexes of most
    important (successfull) articles.
    '''
    days_oct = {}
    days_nov = {}
    selected = days_oct
    offset = len(x_train)

    for index, value in enumerate(target):
        timestamp = complete_test_data.loc[index + offset,
                                           ['contentCreatedFixed']].values
        utc_timestamp = datetime.datetime.utcfromtimestamp(timestamp)
        part_of_day = None
        if (hour):
            part_of_day = utc_timestamp.hour
        else:
            part_of_day = utc_timestamp.day

        if (index == data_oct.shape[0]):
            selected = days_nov

        if not (part_of_day in selected):
            selected[part_of_day] = []
        selected[part_of_day].append((index, value))

    # sort predictions for days based on most predicted total clicks
    for index in days_oct.keys():
        days_oct[index].sort(key=lambda tup: tup[1])  # inplace

    for index in days_nov.keys():
        days_nov[index].sort(key=lambda tup: tup[1])  # inplace

    # choose only x percents articles
    periods = [days_oct, days_nov]
    top_periods = [{}, {}]
    for period_index, period in enumerate(periods):
        for index, value in period.items():
            count = len(value)
            choose = count // (100 // p_choose)

            d = {}
            for i in range(choose):
                d[value[i][0]] = 1
            top_periods[period_index][index] = d
    return (top_periods[0], top_periods[1])


def get_success(items_pred, items_test):
    '''
    Compare most successfull articles in test_set against
    predicted articles.
    '''
    success = {}
    all_good = 0
    total_articles_count = 0
    avg_success = 0
    for index, value in items_test.items():
        good = 0
        articles_count = len(value.keys())
        total_articles_count += articles_count
        for article_id in value.keys():
            if (article_id in items_pred[index]):
                good += 1
        if (articles_count):
            success[index] = round(good / articles_count, 3)
        all_good += good
    avg_success = round((all_good / total_articles_count), 3)
    return (avg_success, success)


def plot_periods(periods, score, month, x_label):
    plt.figure()
    plt.plot(periods, score, 'o-')
    plt.title('{} {}'.format(month, x_label))
    plt.ylabel('Score')
    plt.xlabel(x_label)
    plt.savefig('{}_{}.png'.format(month, x_label))

if __name__ == "__main__":
    scaler = StandardScaler()
    data_sept = pd.read_csv('./data_september_authors.csv')
    data_oct = pd.read_csv('./data_october_authors.csv')
    data_nov = pd.read_csv('./data_november_authors.csv')

    data = pd.concat([data_sept, data_oct, data_nov], ignore_index=True)

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

    # drop index and create new using sorted values by date
    data.reset_index(drop=True, inplace=True)

    # use later to group by days
    complete_test_data = data.iloc[data_sept.shape[0]:data.shape[0], :]

    # DROP FEATURES THAT SHOULD NOT BE USED
    data = data.drop(['day', 'contentAuthor', 'contentCreatedFixed'], axis=1)
    data = data.drop(['contentSection_other',
                      'contentSection_sport',
                      'contentSection_whats-on'], axis=1)
    data = data.drop(['hour_night', 'hour_morning', 'hour_evening'], axis=1)
    data = data.drop(['ref_social'], axis=1)

    y = data['totalClicks']
    x = data.drop(['totalClicks'], axis=1)

    x_train = x.iloc[0:data_sept.shape[0], :]
    x_test = x.iloc[data_sept.shape[0]:data.shape[0], :]

    y_train = y.iloc[0:data_sept.shape[0]].values
    y_test = y.iloc[data_sept.shape[0]:data.shape[0]].values

    # SCALE DATA TO HAVE ZERO MEAN AND UNIT VARIANCE
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # RANDOM FOREST
    pred_train, pred_test = predict_random_forest(x_train, y_train, x_test)

    # SVR
    # pred_train, pred_test = predict_svm(x_train, y_train, x_test)

    # BOOSTED TREE
    # pred_train, pred_test = predict_boosted_tree(x_train, y_train, x_test)

    # ANN
    # pred_train, pred_test = predict_neural(x_train, y_train, x_test)

    # REGRESSION
    # pred_train, pred_test = predict_regression(x_train, y_train, x_test)

    # ROUND ALL VALUES TO INTEGERS
    pred_test = clear_predictions(np.around(pred_test, 0).astype(int), y_test)
    pred_train = clear_predictions(
        np.around(pred_train, 0).astype(int), y_train)

    print_results(y_test, pred_test, y_train, pred_train)
    # plot_results(y_test, pred_test, y_train, pred_train)

    x_per = 10

    top_oct_pred, top_nov_pred = get_most_important_articles(
        pred_test, y_test, x_per,
        complete_test_data, x_train)

    top_oct_test, top_nov_test = get_most_important_articles(
        y_test, y_test, x_per,
        complete_test_data, x_train)

    avg_success_oct, success_oct = get_success(items_pred=top_oct_pred,
                                               items_test=top_oct_test)

    avg_success_nov, success_nov = get_success(items_pred=top_nov_pred,
                                               items_test=top_nov_test)

    print('October')
    # print(success_oct)
    print('Average success', avg_success_oct)

    print('November')
    # print(success_nov)
    print('Average success', avg_success_nov)

    plot_periods(list(success_oct.keys()), list(success_oct.values()),
                 month='October', x_label='Days')
    plot_periods(list(success_nov.keys()), list(success_nov.values()),
                 month='November', x_label='Days')

    top_oct_pred, top_nov_pred = get_most_important_articles(
        pred_test, y_test, x_per,
        complete_test_data, x_train, hour=True)

    top_oct_test, top_nov_test = get_most_important_articles(
        y_test, y_test, x_per,
        complete_test_data, x_train, hour=True)

    avg_success_oct, success_oct = get_success(items_pred=top_oct_pred,
                                               items_test=top_oct_test)

    avg_success_nov, success_nov = get_success(items_pred=top_nov_pred,
                                               items_test=top_nov_test)

    print('October')
    # print(success_oct)
    print('Average success', avg_success_oct)

    print('November')
    # print(success_nov)
    print('Average success', avg_success_nov)

    plot_periods(list(success_oct.keys()), list(success_oct.values()),
                 month='October', x_label='Hours')
    plot_periods(list(success_nov.keys()), list(success_nov.values()),
                 month='November', x_label='Hours')
