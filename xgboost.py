import pandas as pd
from sklearn import linear_model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from pprint import pprint
from time import time
import xgboost as xgb
import cPickle

import numpy as np

np.random.seed(0)

rats_tr = pd.read_csv('data/train.csv')
y = rats_tr.quality
y /= 10
#rats_te = pd.read_csv('data/test.csv')
x = cPickle.load(open('data/train.pkl','rb'))
x = x.as_matrix()
y = y.as_matrix()
kf = KFold(np.shape(x)[0], n_folds=3)
scores = []
max_depthes = [5,6,7]
num_trees = [2000, 2500, 3000, 3500, 4000]

for max_depth in max_depthes:
	score_x = []
	for num_tree in num_trees:
		score_y = []
		print "max_depth, num_tree=%d, %d" %(max_depth, num_tree)
		count=0
		for tr_id, val_id in kf:
			print "round %d" %count
			count+=1
			params = {"objective": "binary:logistic",
				"eta": 0.1,
				"max_depth": max_depth,
				"min_child_weight": 3,
				"silent": 1,
				"subsample": 0.7,
				"colsample_bytree": 0.7,
				"seed": 1}
			x_train, x_val = x[tr_id], x[val_id]
			y_train, y_val = y[tr_id], y[val_id]
			gbm = xgb.train(params, xgb.DMatrix(x_train, y_train), num_tree)
			y_val_bar=gbm.predict(xgb.DMatrix(x_val))
			score_y.append(mean_squared_error(y_val_bar*10, y_val*10)**0.5)
		score_x.append(np.mean(score_y))
		print "mean rmse=%f" %np.mean(score_y)
	scores.append(score_x)

print scores

#max_depthes = [3,4,5,6]
#num_trees = [10,100,1000]
#[0.15642968103337596, 0.16106485139735852, 0.17318190074275239, 
#[1.4957425753263045, 1.5427979680866926, 1.6559345629849074], 
#[1.4906521073883643, 1.5366364218330648, 1.645779840300666], 
#[1.4885068069019813, 1.5324346581555046, 1.6396955685085111]]


#max_depth/num_trees 		2000			2500				3000				3500				4000
#5    			[1.5455126622654021, 1.5448375888402468, 1.546174426277255, 1.5488161644181282, 1.5515069907138905], 
#6    			[1.5459367854480892, 1.5467155499305594, 1.5494475293733236, 1.5529177719342033, 1.5564863900823116],
#7    			[1.5480118290455611, 1.5498062600989584, 1.5535943452350753, 1.5575291854378845, 1.5621487953153184]]

max_depth=5
num_tree=2500
params = {"objective": "binary:logistic",
	"eta": 0.1,
	"max_depth": max_depth,
	"min_child_weight": 3,
	"silent": 1,
	"subsample": 0.7,
	"colsample_bytree": 0.7,
	"seed": 1}
gbm = xgb.train(params, xgb.DMatrix(x, y), num_tree)
ridge = cPickle.load(open('data/model.pkl','rb'))

y_gbm=gbm.predict(xgb.DMatrix(x))
y_ridge=ridge.predict(x)
ensemble=linear_model.LinearRegression()
y_total=np.vstack((y_gbm,y_ridge)).T
ensemble.fit(y_total,y)
ensemble.coef_

test=cPickle.load(open('data/test.pkl','rb'))
ensemble.predict(x_test)

ensemble.fit()