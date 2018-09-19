import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from scipy.sparse import load_npz

xtrain = load_npz('data/xtrain.npz')
xtest = load_npz('data/xtest.npz')
labels = np.load('data/label.npz')

ytrain = labels['train'].reshape(-1, 1)
ytest = labels['test'].reshape(-1, 1)

dtrain = xgb.DMatrix(xtrain, label=ytrain)
dtest = xgb.DMatrix(xtest, label=ytest)

param = {'silent':1, 'objective':'binary:hinge' }
num_round = 500
bst = xgb.train(param, dtrain, num_round)
pred = bst.predict(dtest)
print(classification_report(ytest, pred))
