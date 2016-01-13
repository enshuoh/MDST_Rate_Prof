from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import cPickle
np.random.seed(0)

# Load in the data - pandas DataFrame objects
rats_tr = pd.read_csv('data/train.csv')
y_helpfulness = rats_tr.helpfulness.as_matrix()
y_clarity = rats_tr.clarity.as_matrix()
y_quality = rats_tr.quality.as_matrix()
x = cPickle.load(open('train_x.pkl'))
kf = KFold(x.shape[0], n_folds=3)

alphas = np.power(10.0, np.arange(-2, 8))

for i in range(len(alphas)):
    scores = []
    print "alpha =", alphas[i]
    for train_index, test_index in kf:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y_helpfulness[train_index], y_helpfulness[test_index]
        m1 = linear_model.Ridge(alpha = alphas[i])
        m1.fit(X_train, y_train)
        y_train, y_test = y_clarity[train_index], y_clarity[test_index]
        y1 = m1.predict(X_test)
        m2 = linear_model.Ridge(alpha = alphas[i])
        m2.fit(X_train, y_train)
        y2 = m2.predict(X_test)
        scores.append(mean_squared_error(y1+y2, y_quality[test_index]))
    print np.mean(scores)

y = y_clarity
for i in range(len(alphas)):
    scores = []
    print "alpha =", alphas[i]
    for train_index, test_index in kf:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m = linear_model.Ridge(alpha = alphas[i])
        m.fit(X_train, y_train)
        scores.append(mean_squared_error(m.predict(X_test), y_test))
    print np.mean(scores)
