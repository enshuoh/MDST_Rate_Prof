#
# MDST Ratings Analysis Challenge
# Starter code & ridge regression baseline
#
# Jonathan Stroud
#
#
# Prerequisites:
#
# numpy
# pandas
# sklearn
#

import pandas as pd
# import sklearn.linear_model
from sklearn import linear_model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from pprint import pprint
from time import time

import numpy as np



def grid_search(x, y):
    v = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000,
        max_df=0.8, min_df=50,
        norm = 'l1', use_idf=False)

    v = CountVectorizer(ngram_range=(1, 2), min_df=50)
    x = v.fit_transform(x).todense()
    pipeline = Pipeline([
        ('clf', ExtraTreesRegressor()),
    ])
    # alpha = 10.0 ** np.arange(2, 5)
    parameters = {
        # 'vect__min_df': [10, 20, 40, 100],
        # 'vect__max_df': [0.5, 0.75, 1.0],
        # 'vect__ngram_range': [(1, 2)],  # unigrams or bigrams
        # 'clf__alpha': alpha.tolist(),
        # 'clf__loss': ["squared_loss"],
        'clf__random_state': [0],
        'clf__max_depth': [5, 10, 20, 40, 80],
        'clf__n_estimators': [10, 20, 40, 80, 160],
    }
    grid_search = GridSearchCV(
        pipeline,
        parameters,
        scoring='mean_squared_error',
        n_jobs=-1,
        verbose=3)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(x, y)
    print("done in %0.3fs" % (time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # clf__max_depth: 20
    # clf__n_estimators: 40
    # clf__random_state: 0
    # vect__max_df: 0.75
    # vect__max_features: 2000
    # vect__ngram_range: (1, 2)


def get_result():
    ngram_range = (1, 2)
    max_df = 0.75
    max_features = 2000
    v = CountVectorizer(
        ngram_range=ngram_range,
        max_df=max_df,
        max_features=max_features)
    x = v.fit_transform(rats_tr.comments.fillna('')).todense()
    y = rats_tr.quality
    n_estimators = 40
    max_depth = 20
    clf = ExtraTreesRegressor(n_estimators=n_estimators,
                              max_depth=max_depth,
                              random_state=0)
    clf.fit(x, y)

    t_x = v.transform(rats_te.comments.fillna('')).todense()
    t_y = clf.predict(t_x)
    submit = pd.DataFrame(data={'id': rats_te.id, 'quality': t_y})
    submit.to_csv('ridge_submit.csv', index=False)


def extra_trees_regressor(x, y, n_estimators=10, max_depth=100):
    kf = KFold(len(x), n_folds=3)
    scores = []
    for train_index, test_index in kf:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = ExtraTreesRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  random_state=0)
        clf.fit(X_train, y_train)
        scores.append(mean_squared_error(clf.predict(X_test), y_test) ** 0.5)
    # print 'extraTreeRegressor'
    # print np.mean(scores)
    return np.mean(scores)


def sgd_regressor(x, y, alpha):
    kf = KFold(len(x), n_folds=3)
    scores = []
    for train_index, test_index in kf:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler()
        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)
        clf = SGDRegressor(loss='squared_loss', alpha=alpha)
        clf.fit(x_train, y_train)
        scores.append(mean_squared_error(clf.predict(x_test), y_test) ** 0.5)
    # print 'SGDRegressor'
    return np.mean(scores)
    # return scores


def tune_sgd(x, y):
    # alpha=335160, score = 0.718329
    alpha = 10.0 ** np.arange(-1, 7)
    for a in alpha:
        print 'alpha = %f' % a
        print 'score = %f' % sgd_regressor(x, y, a)

def test_lasso_regressior(x, y, alpha):
    kf = KFold(len(x), n_folds=3)
    scores = []
    coef = []
    count = 0
    for train_index, test_index in kf:
        print count
        count+=1
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(X_train, y_train)
        scores.append(mean_squared_error(clf.predict(X_test), y_test) ** 0.5)
        coef.append(clf.coef_)
    # print 'extraTreeRegressor'
    # print np.mean(scores)
    return scores, coef


def tune_lasso(x, y, alpha=10.0 ** np.arange(-4, -2)):
    coefs = []
    scores = []
    for a in alpha:
        score, coef = test_lasso_regressior(x, y, a)
        coefs.append(coef)
        scores.append(score)
        print 'alpha=%f, score = %f' % (a, np.mean(scores))
    return scores, coefs

def select_feature(x, y):
    v = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000,
        max_df=0.8, min_df=50,
        norm = 'l1', use_idf=False)
    x = v.fit_transform(x).todense()
    clf = linear_model.Lasso(alpha=10e-4)
    clf.fit(x, y)
    idx = sorted(range(len(clf.coef_)), key=lambda k: -abs(clf.coef_[k]))
    columns = v.get_feature_names()
    vocabulary = []
    for i in idx:
        if abs(clf.coef_[i]) == 0:
            break
        vocabulary.append(columns[i])
    return vocabulary

np.random.seed(0)

# Load in the data - pandas DataFrame objects

rats_tr = pd.read_csv('data/train.csv')
rats_te = pd.read_csv('data/test.csv')
# X_train, X_test, y_train, y_test = train_test_split(
#     rats_tr.comments.fillna(''), rats_tr.quality, test_size=0.33, random_state=0)
x, X_test, y, y_test = train_test_split(
    rats_tr.comments.fillna(''), rats_tr.quality,
    test_size=0.7, random_state=42)
x = x.as_matrix()
y = y.as_matrix()

vocabulary = select_feature(x, y)

v = TfidfVectorizer(vocabulary=vocabulary, use_idf=False)
x = v.transform(rats_tr.comments.fillna(''))
y = rats_tr.quality


grid_search(rats_tr.comments.fillna(''), rats_tr.quality)


# kf = KFold(len(x), n_folds=5)
# scores = []
# for train_index, test_index in kf:
#     X_train, X_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     clf = ExtraTreesRegressor(n_estimators=10,
#                               max_depth=100,
#                               random_state=0)
#     clf.fit(X_train, y_train)
#     scores.append(mean_squared_error(clf.predict(X_test), y_test) ** 0.5)
# # print 'extraTreeRegressor'
# # print np.mean(scores)

# print np.mean(scores)

