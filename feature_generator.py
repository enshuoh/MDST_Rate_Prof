import pandas as pd
import numpy as np
from textblob import TextBlob


rats_tr = pd.read_csv('data/train.csv')
map_dict = {
    'forcredit': {np.nan: 0, 'Yes': 1, 'No': -1},
    'attendance': {np.nan: 0, 'Not Mandatory': 1, 'Mandatory': -1},
}
dummy = ['textbookuse', 'interest', 'grade']
for key in map_dict:
    rats_tr[key] = rats_tr[key].map(map_dict[key])
for key in dummy:
    rats_tr = pd.concat((rats_tr, pd.get_dummies(rats_tr[key])), 1)
    rats_tr = rats_tr.drop(key)
rats_tr['tags'] = rats_tr.tags.map(
    eval).str.join(sep="\t").str.get_dummies(sep="\t")