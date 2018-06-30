#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys
# sys.path.insert(0,'/Users/joaoschubnell/Documents/fineFoods')

import pandas as pd
import sklearn as sk
import numpy
from time import time

colnames = ["product/productId",
            "review/userId",
            "review/helpfulness",
            "review/score",
            "review/time",
            "review/summary",
            "review/text"]

df = pd.read_csv("data/finemuged.csv", encoding="latin1", header=None,
                 names=colnames, quotechar = "\"")

df["review/date"] = pd.to_datetime(df["review/time"], unit="s")

df = df[df["review/summary"].notnull()]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from scipy.sparse import csr_matrix

vect = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
X_tfidf = vect.fit_transform(df['review/summary'].values)
X = csr_matrix(X_tfidf)
y = df['review/score'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)



clf = Pipeline([
    ('PAC', PassiveAggressiveClassifier())
])

t0 = time()
clf.fit(X_train, y_train)
print("PassiveAggressive done in %fs" % (time() - t0))

y_predict = clf.predict(X_test)
print("PassiveAggressive")
print(classification_report(y_test, y_predict))



clf = Pipeline([
    ('SVC', LinearSVC(class_weight='balanced'))
])

t0 = time()
clf.fit(X_train, y_train)
print("LinearSVC done in %fs" % (time() - t0))

y_predict = clf.predict(X_test)
print("LinearSVC")
print(classification_report(y_test, y_predict))



clf = Pipeline([
    ('LR', LogisticRegression(class_weight='balanced'))
])

t0 = time()
clf.fit(X_train, y_train)
print("LogisticRegression done in %fs" % (time() - t0))

y_predict = clf.predict(X_test)
print("LogisticRegression")
print(classification_report(y_test, y_predict))
