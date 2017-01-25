import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sklearn.ensemble as dt
import itertools
import sklearn.linear_model as lr
import sklearn.neural_network as nn
import sklearn.svm as svm
from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

FIGURES = './figures/'
DATA_ANALYSIS = '{}data_analysis/'.format(FIGURES)
RESULTS = '{}results/'.format(FIGURES)
TRANSFORMED_DATA = './transformed_data/'
CV_END_POSITION = 200


class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def create_dirs():
    if not os.path.exists(TRANSFORMED_DATA):
        os.makedirs(TRANSFORMED_DATA)
    if not os.path.exists(FIGURES):
        os.makedirs(FIGURES)
    if not os.path.exists(DATA_ANALYSIS):
        os.makedirs(DATA_ANALYSIS)
    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def color_print(text, color=Colors.GREEN):
    print(color + text + Colors.ENDC)


def to_numeric(data):
    '''
    Converts all string values into integer values
    '''
    dt = dict(data.dtypes)
    categories = []
    for key, value in dt.items():
        if (value == 'object'):
            categories.append(key)

    for col in categories:
        data[col] = data[col].astype('category')

    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    return data


def init_figure():
    plt.figure(figsize=(35, 15))
    plt.ion()  # set interactive mode on


def create_box_plot(data, img_name):
    data.boxplot()
    plt.savefig('{}{}.png'.format(DATA_ANALYSIS, img_name))


def create_hist(data, img_name):
    data.hist(figsize=(35, 15))
    plt.savefig('{}{}.png'.format(DATA_ANALYSIS, img_name))


def save_data(data, name):
    data.to_csv('{}{}.csv'.format(TRANSFORMED_DATA, name))


def create_correlation_matrix(data, img_name):
    corr = data.corr()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, interpolation='nearest')
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('{}{}.png'.format(DATA_ANALYSIS, img_name))


def print_shape_info(data_train, data_cv, data_test):
    color_print('SHAPE INFO:')
    print('Train data', data_train.shape)
    print('Cross validation data', data_cv.shape)
    print('Test data', data_test.shape)


def describe_properties(data_train, data_cv_test):
    color_print('Train data properties:')
    print(data_train.describe())
    color_print('CV and Test data properties:')
    print(data_cv_test.describe())


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    This code is taken from sklearn examples.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def predict_logistic_regression(x_train, y_train, x_test, y_test, predicting):
    '''
    Do not use validation set here because I do not perform
    any parameter search for logistic regression. Logistic
    regression here is only used as baseline.
    '''
    l = lr.LogisticRegression()
    l.fit(x_train, y_train.values.ravel())  # reshape
    y_pred = l.predict(x_test)

    color_print('Logistic regression')
    print('accuracy', accuracy_score(y_test, y_pred))
    print('f-score (global)', f1_score(y_test, y_pred, labels=[1, 2, 3, 4, 5],
                                       average='micro'))
    print('f-score (local)', f1_score(y_test, y_pred, labels=[1, 2, 3, 4, 5],
                                      average='macro'))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[1, 2, 3, 4, 5],
                          title='Confusion matrix')
    plt.savefig('{}logistic_regression_{}.png'.format(RESULTS, predicting))


def predict_random_forest(x_train, y_train, x_cv, y_cv, x_test, y_test,
                          predicting):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_cv, y_cv, x_test, y_test)

    rf = dt.RandomForestClassifier()
    '''
    parameters = {
        'n_estimators': [60, 80, 100, 120],
        'max_depth': [5, 10, 15, 20, 25, 30]
    }
    '''
    parameters = {
        'n_estimators': [120], 'max_depth': [20]
    }
    # best_params = clf.best_params_
    best_params = None
    clf = GridSearchCV(rf, parameters, scoring=score, cv=ps)
    clf.fit(x, y.values.ravel())
    y_pred = clf.predict(x_test)

    eval_learning('Random forest', y_test, y_pred, best_params,
                  predicting)


def get_grid_search_data(x_train, y_train, x_cv, y_cv, x_test, y_test):
    # allows to use validation set instead of cross-validation
    # using GridSearchCV
    index_train = [-1] * len(x_train)
    index_validate = [0] * len(x_cv)
    indexes = index_train + index_validate
    ps = PredefinedSplit(test_fold=indexes)

    x = x_train.append(x_cv, ignore_index=True)
    y = y_train.append(y_cv, ignore_index=True)
    score = 'f1_macro'  # same weight for all classes

    return (x, y, ps, score)


def predict_neural_network(x_train, y_train, x_cv, y_cv, x_test, y_test,
                           predicting):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_cv, y_cv, x_test, y_test)

    nm = nn.MLPClassifier()
    '''
    parameters = {
        'hidden_layer_sizes': [[50], [75], [100], [50, 50], [75, 75],
                               [100, 100], [50, 50, 50], [75, 75, 75],
                               [100, 100, 100], [100, 75, 50]],
        'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    }
    '''
    parameters = {
        'hidden_layer_sizes': [75, 75], 'alpha': [0.01]
    }
    # best_params = clf.best_params_
    best_params = None
    clf = GridSearchCV(nm, parameters, scoring=score, cv=ps)
    clf.fit(x, y.values.ravel())
    y_pred = clf.predict(x_test)

    eval_learning('Neural network', y_test, y_pred,
                  best_params, predicting)


def predict_svm(x_train, y_train, x_cv, y_cv, x_test, y_test, predicting):
    x, y, ps, score = get_grid_search_data(
        x_train, y_train, x_cv, y_cv, x_test, y_test)

    svc = svm.SVC()

    # comment out to perform grid search
    '''
    parameters = [
        {'C': [1, 10, 30, 100, 300, 1000],
            'kernel': ['poly'], 'degree': [2, 3, 4]},
    ]
    '''
    parameters = [
        {'C': [100], 'kernel': ['poly'], 'degree': [2]},
    ]

    clf = GridSearchCV(svc, parameters, scoring=score, cv=ps)
    clf.fit(x, y.values.ravel())
    y_pred = clf.predict(x_test)

    # best_params = clf.best_params_
    best_params = None
    eval_learning('SVM', y_test, y_pred, best_params, predicting)


def eval_learning(name, y_test, y_pred, best_params, predicting):
    color_print(name)
    labels = [1, 2, 3, 4, 5]

    if (best_params):
        print('Best parameters found on validation set:')
        print(best_params)

    print('Accuracy', accuracy_score(y_test, y_pred))
    print('F-score (global, micro)', f1_score(y_test, y_pred,
                                              labels=labels,
                                              average='micro'))
    print('F-score (local, macro)', f1_score(y_test, y_pred,
                                             labels=labels,
                                             average='macro'))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels,
                          title='Confusion matrix')
    plt.savefig('{}{}_{}.png'.format(RESULTS, name.lower().replace(' ', '_'),
                                     predicting))


def get_most_important_features(x_train, y_train, x_cv, y_cv, x_test, y_test,
                                predicting):
    x = x_train.append(x_cv, ignore_index=True).append(
        x_test, ignore_index=True)
    y = y_train.append(y_cv, ignore_index=True).append(
        y_test, ignore_index=True)
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
    plt.savefig('{}{}_{}.png'.format(
        DATA_ANALYSIS, 'feature_importances', predicting))


def pca_reduction(data):
    data = preprocessing.scale(data)
    pca = PCA(n_components=data.shape[1])
    pca.fit_transform(data)

    explained_variance_ratio = pca.explained_variance_ratio_ * 100
    cumulated = np.cumsum(explained_variance_ratio)

    y = [i for i in range(1, data.shape[1] + 1)]

    plt.figure()
    plt.subplot(211)
    plt.plot(y, explained_variance_ratio, 'k')
    plt.ylim(0, 100)
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained %')
    plt.subplot(212)
    plt.plot(y, cumulated, 'k')
    plt.ylim(0, 100)
    plt.xlabel('Principal component')
    plt.ylabel('Cumulative variance explained')
    plt.tight_layout()
    plt.savefig('{}pca.png'.format(DATA_ANALYSIS))


def divide_dataset(drop, y):
    '''
    Divide to Train, CV and Test sets
    '''
    x_train = data_train.drop(drop, axis=1)
    y_train = data_train[y]

    x_cv = data_cv.drop(drop, axis=1)
    y_cv = data_cv[y]

    x_test = data_test.drop(drop, axis=1)
    y_test = data_test[y]

    return (x_train, y_train, x_cv, y_cv, x_test, y_test)

if __name__ == '__main__':
    clear_console()
    create_dirs()
    init_figure()
    pd.set_option('display.max_columns', None)

    # LOAD DATA
    data_train = to_numeric(pd.read_csv('./data/student-por.csv'))
    data_cv_test = to_numeric(pd.read_csv('./data/student-mat.csv'))
    data_cv = data_cv_test.iloc[0:CV_END_POSITION, :]
    data_test = data_cv_test.iloc[CV_END_POSITION:data_cv_test.shape[0], :]

    # SAVE TRANSFORMED DATA (string features to numeric)
    save_data(data_train, 'train_data_numeric')
    save_data(data_cv_test, 'cv_test_data_numeric')

    # DATA ANALYSIS PART
    print_shape_info(data_train, data_cv, data_test)
    describe_properties(data_train, data_cv_test)

    create_box_plot(data=data_train, img_name='train_data_box_plot')
    create_box_plot(data=data_cv_test, img_name='cv_and_test_data_box_plot')

    create_hist(data=data_train, img_name='train_data_hist')
    create_hist(data=data_cv_test, img_name='cv_and_test_data_hist')

    create_correlation_matrix(
        data=data_train, img_name='train_data_correlation')
    create_correlation_matrix(
        data=data_cv_test, img_name='cv_and_data_correlation')

    # Create dummies variables
    dummies_columns = ['Mjob', 'Fjob', 'reason', 'guardian']
    data_train = pd.get_dummies(
        data_train, columns=dummies_columns, drop_first=True)
    data_cv_test = pd.get_dummies(
        data_cv_test, columns=dummies_columns, drop_first=True)
    data_cv = data_cv_test.iloc[0:CV_END_POSITION, :]
    data_test = data_cv_test.iloc[CV_END_POSITION:data_cv_test.shape[0], :]

    print_shape_info(data_train, data_cv, data_test)

    # SAVE TRANSFORMED DATA (Add dummie variables)
    save_data(data_train, 'train_data_numeric_dummies')
    save_data(data_cv_test, 'cv_test_data_numeric_dummies')

    # Display PCA plot
    pca_reduction(data_train.append(data_cv_test))

    drop_list = [['Dalc', 'Walc', 'G1', 'G2', 'G3'],
                 ['Dalc', 'Walc', 'G1', 'G2', 'G3'],
                 ['G1', 'G2', 'G3', 'health']]
    y_list = [['Dalc'], ['Walc'], ['health']]
    y_name_list = ['Alkohol_day', 'Alkohol_weekend', 'Health']

    # Predict alkohol comsumption for week, weekend and health status
    for i in range(len(drop_list)):
        x_train, y_train, x_cv, y_cv, x_test, y_test = divide_dataset(
            drop=drop_list[i], y=y_list[i])

        # IDENTIFY MOST IMPORTANT FEATURES
        get_most_important_features(x_train, y_train, x_cv, y_cv, x_test,
                                    y_test, predicting=y_name_list[i])

        # BASELINE MODEL (logistic regression)
        predict_logistic_regression(x_train=x_train, y_train=y_train,
                                    x_test=x_test, y_test=y_test,
                                    predicting=y_name_list[i])

        # Random forest
        predict_random_forest(x_train=x_train, y_train=y_train, x_test=x_test,
                              y_test=y_test, x_cv=x_cv, y_cv=y_cv,
                              predicting=y_name_list[i])

        # Neural network
        predict_neural_network(x_train=x_train, y_train=y_train, x_test=x_test,
                               y_test=y_test, x_cv=x_cv, y_cv=y_cv,
                               predicting=y_name_list[i])

        # SVM
        predict_svm(x_train=x_train, y_train=y_train, x_test=x_test,
                    y_test=y_test, x_cv=x_cv, y_cv=y_cv,
                    predicting=y_name_list[i])
