#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.insert(0,'/Users/joaoschubnell/Documents/fineFoods')

import pandas as pd
import sklearn as sk
import numpy

# with open("data/finefoods-cleaned.txt", "w+") as fout:
#     with open("data/finefoods.txt", encoding="latin1") as file:
#         for x in file.readlines():
#             fout.write(x.replace("\"",""))
# 
# with open("data/finemuged.csv", "w+") as fout:
#     print("",file=fout)
#     with open("data/finefoods-cleaned.txt", encoding="latin1") as file:
#         for x in file.readlines():
#             l = x.split(": ")
#             if x == "\n":
#                 pass
#             elif x.startswith("product/productId"):
#                 print("\""+l[1].strip("\n")+"\"",
#                       file=fout, end=",")
#             elif x.startswith("review/userId"):
#                 print("\""+l[1].strip("\n")+"\"",
#                       file=fout, end=",")
#             elif x.startswith("review/helpfulness"):
#                 print("\""+l[1].strip("\n")+"\"",
#                       file=fout, end=",")
#             elif x.startswith("review/score"):
#                 print("\""+l[1].strip("\n")+"\"",
#                       file=fout, end=",")
#             elif x.startswith("review/time"):
#                 print("\""+l[1].strip("\n")+"\"",
#                       file=fout, end=",")
#             elif x.startswith("review/summary"):
#                 print("\""+l[1].strip("\n")+"\"",
#                       file=fout, end=",")
#             elif x.startswith("review/text"):
#                 print("\""+l[1].strip("\n")+"\"",
#                       file=fout)
# 
# 
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
stopw = []
with open("data/stop_word.txt") as f:
    for x in f.readlines():
        stopw.append(x.strip("\n"))

print(df[df["review/summary"].isnull()]["review/text"])

df = df[df["review/summary"].notnull()]

countvect = CountVectorizer(stop_words=stopw)
review_text = countvect.fit_transform(df["review/text"].values.astype('U'))
review_summary = countvect.fit_transform(df["review/summary"].values.astype('U'))
# print(review_summary.shape)
print(df.info())

df.to_csv('/Users/joaoschubnell/Documents/fineFoods/data/reviews.csv')

numpy.savetxt('Users/joaoschubnell/Documents/fineFoods/data/review_text.csv', review_text.torray(), delimiter = ',')

numpy.savetxt('Users/joaoschubnell/Documents/fineFoods/data/review_summary.csv', review_summary.toarray(), delimiter = ',')