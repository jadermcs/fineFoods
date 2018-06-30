#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys
# sys.path.insert(0,'/Users/joaoschubnell/Documents/fineFoods')

import pandas as pd
import sklearn as sk
import numpy

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
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

X = df['review/summary'].values
y = df['review/score'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

vect = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

clf = Pipeline([
    ('tfidf', vect),
    ('MNB', MultinomialNB())
])

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print("MultinomialNB")
print(classification_report(y_test, y_predict))

clf = Pipeline([
    ('tfidf', vect),
    ('SVC', LinearSVC())
])

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print("LinearSVC")
print(classification_report(y_test, y_predict))

exit()

clf = Pipeline([
    ('tfidf', vect),
    ('XGB', XGBClassifier(n_jobs=4))
])

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print("XGB")
print(classification_report(y_test, y_predict))

clf = Pipeline([
    ('tfidf', vect),
    ('MLayerP', MLPClassifier(n_jobs=4))
])

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print("MLayerP")
print(classification_report(y_test, y_predict))
