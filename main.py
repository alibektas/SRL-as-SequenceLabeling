
from pathlib import Path
import sys
import os 
import spacy 
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt
import json
import pandas

from semantictagger.dataset import Dataset
from semantictagger.datastats import collectaccuracy
from semantictagger.paradigms import BNE , GLOB , LOCTAG , DIRECTTAG
from semantictagger.conllu import CoNLL_U



fp = Path("./UP_English-EWT/en_ewt-up-train.conllu")
dataset = Dataset(fp)


dirtag = DIRECTTAG(1)
dirtag2 = DIRECTTAG(2)
collectaccuracy(dirtag , dataset , showresults=True)
sparsity , mean , std , dict_ = dataset.getlabelfrequencies(dirtag , show_results=True , returndict = True)

