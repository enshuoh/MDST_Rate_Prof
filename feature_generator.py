import pandas as pd
import numpy as np
import cPickle
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def gen_feature(csv_file='data/train.csv'):
    rats_tr = pd.read_csv(csv_file)
    map_dict = {
        'forcredit': {np.nan: 0, 'Yes': 1, 'No': -1},
        'attendance': {np.nan: 0, 'Not Mandatory': 1, 'Mandatory': -1},
    }
    dummy = ['textbookuse', 'interest', 'grade']
    for key in map_dict:
        rats_tr[key] = rats_tr[key].map(map_dict[key])

    for key in dummy:
        rats_tr = pd.concat((rats_tr, pd.get_dummies(rats_tr[key])), 1)
        rats_tr = rats_tr.drop(key, 1)

    rats_tr['tags'] = rats_tr.tags.map(eval).str.join(sep="\t").str.get_dummies(sep="\t")
    rats_tr = rats_tr.drop('tags', 1)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=10, use_idf=True)
    trans_x = vectorizer.fit_transform(rats_tr.comments.fillna(''))

    helpfulness = rats_tr.helpfulness
    clarity = rats_tr.clarity
    quality = rats_tr.quality

    drop_columns = ['id', 'tid', 'date', 'comments',
                    'online', 'helpfulness', 'clarity', 'quality']
    rats_tr = rats_tr.drop(drop_columns, 1)

    train_x = sparse.hstack((sparse.csr_matrix(rats_tr.as_matrix()), trans_x), format='csr')
    cPickle.dump(train_x, open(csv_file.replace('.csv', '.pkl'), 'w'))
