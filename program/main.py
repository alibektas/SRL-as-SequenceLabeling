
from pathlib import Path
import sys
import os 
import spacy 
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt
import json
import pandas

from dataset import Dataset
from datastats import collectaccuracy
from paradigms import BNE , GLOB , LOCTAG , DIRECTTAG
from conllu import CoNLL_U



fp = Path("../UP_English-EWT/en_ewt-up-train.conllu")
dataset = Dataset(fp)


def brand_new_stats():
    dict_ = {}
    for i in dataset.entries:
        dist = i.brand_new_encoding()
        for j in dist:       
            if j in dict_:
                dict_[j] += 1
            else :
                dict_[j] = 1
    collect_stats(dict_)


def bio(labels):
    begin = False
    bios_tags = [""] * len(labels)
    for index , value in enumerate(labels):
        if value.startswith("0"):
            bios_tags[index] = "B"
            begin = True

        elif value == "":
            if begin:
                bios_tags[index] = "O"
                begin = False
            else:
                continue
        else:
            if begin:
                bios_tags[index] = "I"
            
    return bios_tags



dirtag = DIRECTTAG(3)
dirtag2 = DIRECTTAG(3)
collectaccuracy(dirtag , dataset)
sparsity , mean , std , dict_ = dataset.getlabelfrequencies(dirtag , returndict = True)

print(dict_)