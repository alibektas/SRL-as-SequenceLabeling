import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim as optim
import torch.nn.utils as util

import flair
from flair.data import Sentence 

import sys 
import spacy 
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt
import json

import ccformat
import numpy as np 
from os import path
from pathlib import Path
from semantictagger.dataset import Dataset
from semantictagger.paradigms import DIRECTTAG


from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings , ELMoEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer



# curdir = path.dirname(__file__)

# def createcolumncorpusfiles():
#     train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
#     test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
#     dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
#     dataset_train = Dataset(train_file)
#     dataset_test = Dataset(test_file)
#     dataset_dev = Dataset(dev_file)

#     dirtag = DIRECTTAG(2 , verbshandler='omitverb' , verbsonly=False)
#     verbsencoder = DIRECTTAG(2 , verbshandler='omitsense' , verbsonly=True)

#     ccformat.writecolumncorpus(dataset_train , dirtag, filename="train" , verbmarker = True , verbsonlyencoder=verbsencoder)
#     ccformat.writecolumncorpus(dataset_dev , dirtag, filename="dev" , verbmarker = True, verbsonlyencoder=verbsencoder)
#     ccformat.writecolumncorpus(dataset_test , dirtag, filename="test" , verbmarker=True,verbsonlyencoder=verbsencoder)

# if not path.isfile(path.join(curdir,"data","train.txt")):
#         print("Data not found.")
#         createcolumncorpusfiles()
# else :
#     print("Training data exist.")
    
# # define columns
# columns = {0: 'text', 1: 'srl'}

# # init a corpus using column format, data folder and the names of the train, dev and test files
# corpus: Corpus = ColumnCorpus(path.join(curdir,"data"),
#                             columns,
#                             train_file='train.txt',
#                             test_file='test.txt',
#                             dev_file='dev.txt')



# tag_type = 'srl'
# tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# 4. initialize embeddings
embedding_types = [
    ELMoEmbeddings('original'),
    ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

s1 = Sentence("is")
s2 = Sentence("is*") 

embeddings.embed(s1)
embeddings.embed(s2)

for i in range(len(s1)):
    print(s1[0].embedding)
    print(s2[0].embedding)


# tagger: SequenceTagger = SequenceTagger(
#         hidden_size=512,
#         embeddings=embeddings,
#         tag_dictionary=tag_dictionary,
#         rnn_layers = 2,                                                                 tag_type=tag_type,
#         use_crf=True)

# trainer: ModelTrainer = ModelTrainer(tagger, corpus)
# # 7. start training
# trainer.train(path.join(curdir,"modelout"),
#                     learning_rate=0.1,
#                     mini_batch_size=32,
#                     embeddings_storage_mode="gpu",
#                     max_epochs=150)

