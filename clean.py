#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys
sys.path.insert(0,'/Users/joaoschubnell/Documents/fineFoods')

import pandas as pd
import sklearn as sk
import numpy

with open("data/finefoods-cleaned.txt", "w+") as fout:
    with open("data/finefoods.txt", encoding="latin1") as file:
        for x in file.readlines():
            fout.write(x.replace("\"",""))

with open("data/finemuged.csv", "w+") as fout:
    print("",file=fout)
    with open("data/finefoods-cleaned.txt", encoding="latin1") as file:
        for x in file.readlines():
            l = x.split(": ")
            if x == "\n":
                pass
            elif x.startswith("product/productId"):
                print("\""+l[1].strip("\n")+"\"",
                      file=fout, end=",")
            elif x.startswith("review/userId"):
                print("\""+l[1].strip("\n")+"\"",
                      file=fout, end=",")
            elif x.startswith("review/helpfulness"):
                print("\""+l[1].strip("\n")+"\"",
                      file=fout, end=",")
            elif x.startswith("review/score"):
                print("\""+l[1].strip("\n")+"\"",
                      file=fout, end=",")
            elif x.startswith("review/time"):
                print("\""+l[1].strip("\n")+"\"",
                      file=fout, end=",")
            elif x.startswith("review/summary"):
                print("\""+l[1].strip("\n")+"\"",
                      file=fout, end=",")
            elif x.startswith("review/text"):
                print("\""+l[1].strip("\n")+"\"",
                      file=fout)
