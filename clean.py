#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys
from tqdm import tqdm

import pandas as pd
import sklearn as sk
import numpy

with open("data/finefoods-cleaned.txt", "w+") as fout:
    with open("data/finefoods.txt", encoding="latin1") as file:
        print("Cleaning file...")
        for x in tqdm(file.readlines(), ascii='#', ncols=100):
            fout.write(x.replace("\"",""))

with open("data/finemuged.csv", "w+") as fout:
    print("",file=fout)
    with open("data/finefoods-cleaned.txt") as file:
        print("Converting format to csv...")
        for x in tqdm(file.readlines(), ascii='#', ncols=100):
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
