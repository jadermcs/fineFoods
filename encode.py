#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import pandas as pd
import numpy as np
import scipy as sci

colnames = ["product/productId",
            "review/userId",
            "review/helpfulness",
            "review/score",
            "review/time",
            "review/summary",
            "review/text"]

df = pd.read_csv("data/finemuged.csv", encoding="latin1", header=None,
                 names=colnames, quotechar = "\"").sample(100000)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["review/date"] = pd.to_datetime(df["review/time"], unit="s")

print(df[df["review/score"] == 2]["review/text"].sample(3))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.externals import joblib
import spacy

nlp = spacy.load('en_core_web_sm')

vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2),
                       max_features=5000)
X = vect.fit_transform(df['review/text'].values)
# y = np.where(df['review/score'].values > 2, 1, 0)
y = df['review/score'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

save_npz('data/xtrain.npz', X_train)
save_npz('data/xtest.npz', X_test)

np.savez('data/label.npz', train=y_train, test=y_test)
