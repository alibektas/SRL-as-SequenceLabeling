import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim as optim
import torch.nn.utils as util

import flair
from flair.embeddings import FlairEmbeddings as FE
from flair.data import Sentence 

from pathlib import Path
import sys
import os 
import spacy 
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt
import json
import pandas

import ccformat

from semantictagger.dataset import Dataset
from semantictagger.paradigms import DIRECTTAG

import numpy as np 


def createcolumncorpusfiles():
    train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
    test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
    dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
    dataset_train = Dataset(train_file)
    dataset_test = Dataset(test_file)
    dataset_dev = Dataset(dev_file)

    dirtag = DIRECTTAG(3)

    ccformat.writecolumncorpus(dataset_train , dirtag)
    ccformat.writecolumncorpus(dataset_dev , dirtag)
    ccformat.writecolumncorpus(dataset_test , dirtag)


from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: 'text', 1: 'srl'}

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus('/Users/alibektas/Documents/ThesisDev/cur/data',
                            columns,
                            train_file='train.txt',
                            test_file='test.txt',
                            dev_file='dev.txt')

