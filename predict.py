#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from scipy.sparse import load_npz, hstack
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

X_train = load_npz('data/xtrain.npz')
X_test = load_npz('data/xtest.npz')

labels = np.load('data/label.npz')
y_train = labels['train']
y_test = labels['test']

clf = Pipeline([
    ('SVD', TruncatedSVD(100)),
    ('PAC', PassiveAggressiveClassifier(class_weight='balanced'))
])

params = {
	'PAC__C': np.logspace(0, 3, 4)
}

grid = GridSearchCV(
	clf,
	params,
	n_jobs=2,
        verbose=1,
	cv=StratifiedKFold(5)
)

t0 = time()
grid.fit(X_train, y_train)
print("PassiveAggressive done in %fs" % (time() - t0))

train_predict = grid.predict(X_train)
test_predict = grid.predict(X_test)
print("PassiveAggressive")
print(classification_report(y_test, test_predict))
# clf = PassiveAggressiveClassifier(class_weight='balanced')
# clf.set_params(grid.best_params_)
# joblib.dump(clf, 'pac.pkl')


clf = Pipeline([
    ('SVD', TruncatedSVD(100)),
    ('SVC', LinearSVC(class_weight='balanced'))
])

params = {
	'SVC__C': np.logspace(0, 3, 4)
}

grid = GridSearchCV(
	clf,
	params,
	n_jobs=2,
        verbose=2,
	cv=StratifiedKFold(5)
)
t0 = time()
grid.fit(X_train, y_train)
print("LinearSVC done in %fs" % (time() - t0))

train_predict = grid.predict(X_train)
test_predict = grid.predict(X_test)
print("LinearSVC")
print(classification_report(y_test, test_predict))
# joblib.dump(grid, 'svm.pkl')


clf = Pipeline([
    ('SVD', TruncatedSVD(100)),
    ('LR', LogisticRegression(class_weight='balanced'))
])

params = {
	'LR__C': np.logspace(0, 3, 4)
}

grid = GridSearchCV(
	clf,
	params,
        n_jobs=2,
        verbose=2,
	cv=StratifiedKFold(5)
)

t0 = time()
grid.fit(X_train, y_train)
print("LogisticRegression done in %fs" % (time() - t0))

train_predict = grid.predict(X_train)
test_predict = grid.predict(X_test)
print("LogisticRegression")
print(classification_report(y_test, test_predict))
# joblib.dump(grid, 'logit.pkl')
