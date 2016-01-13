from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer,TfidfVectorizer
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# Load in the data - pandas DataFrame objects
rats_tr = pd.read_csv('data/train.csv')
rats_te = pd.read_csv('data/test.csv')


vect_list = [
    TfidfVectorizer(ngram_range=(1, 3), min_df=10,use_idf=True),
    TfidfVectorizer(ngram_range=(1, 2), min_df=20 ,use_idf=True),
    TfidfVectorizer(ngram_range=(1, 2), min_df=40 ,use_idf=True),
    TfidfVectorizer(ngram_range=(1, 2), min_df=80 ,use_idf=True),
    TfidfVectorizer(ngram_range=(1, 2), max_df=0.8,min_df=10,use_idf=True),
    TfidfVectorizer(ngram_range=(1, 2), max_df=0.6,min_df=20 ,use_idf=True),
    TfidfVectorizer(ngram_range=(1, 2), max_df=0.4,min_df=40 ,use_idf=True),
    TfidfVectorizer(ngram_range=(1, 2), max_df=0.2,min_df=80 ,use_idf=True),
]

y = Ytrain = np.ravel(rats_tr.quality)
for idx, vect in enumerate(vect_list):
    # Construct bigram representation
    # "Fit" the transformation on the training set and apply to test
    x = vect.fit_transform(rats_tr.comments.fillna(''))
    kf = KFold(x.shape[0], n_folds=3)

    alphas = np.power(10.0, np.arange(-1, 2))
    mseTr = np.zeros((len(alphas),))
    mseVal = np.zeros((len(alphas),))

    # Search for lowest validation accuracy
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


# alpha = 0.1
# 3.76435595383
# alpha = 1.0
# 2.84146459484
# alpha = 10.0
# 2.98464617608
# alpha = 0.1
# 3.47087185031
# alpha = 1.0
# 2.85202073447
# alpha = 10.0
# 2.94833810099
# alpha = 0.1
# 3.17004435697
# alpha = 1.0
# 2.87278954982
# alpha = 10.0
# 2.92819652333
# alpha = 0.1
# 3.00733177055
# alpha = 1.0
# 2.89238903398
# alpha = 10.0
# 2.92796413364
# alpha = 0.1
# 3.76435595383
# alpha = 1.0
# 2.84146459484
# alpha = 10.0
# 2.98464617608
# alpha = 0.1
# 3.47448976899
# alpha = 1.0
# 2.85356247499
# alpha = 10.0
# 2.9473462559
# alpha = 0.1
# 3.17446213489
# alpha = 1.0
# 2.87662036839
# alpha = 10.0
# 2.92679042024
# alpha = 0.1
# 3.02080223545
# alpha = 1.0
# 2.90753496343
# alpha = 10.0
# 2.93687811219
