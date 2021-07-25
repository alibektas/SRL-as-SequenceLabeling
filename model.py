from flair.embeddings import WordEmbeddings
from flair.data import Sentence

import torch
import math 

import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F

from pathlib import Path
import os
from semantictagger import paradigms 
from semantictagger.paradigms import DIRECTTAG, Encoder, PairMapper , RELPOS , MapProtocol , Mapper
from semantictagger.dataset import Dataset
import ccformat


import logging
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

DEVELOPMENT = True
INFEATURES = d_model = 100
curdir = os.path.dirname(__file__)
path_to_data = os.path.join(curdir , "data")
train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
train = Dataset(train_file)
test = Dataset(test_file)
dev = Dataset(dev_file)

datasets = [train , test , dev]

pm = PairMapper(
        roles = 
        [
            ("ARG1","V","XARG1V"),
            ("ARG2","V","XARG2V"),
            ("V","ARG2","XARG2V"),
            ("V","ARG1","XARG1V"),
            ("ARG1","ARG0","XARG1-0"),
            #("ARG0","ARG0","XARG0-0"),
        ]
    )
tagger = DIRECTTAG(2,verbshandler="omitlemma",deprel=True,depreldepth=3 , pairmap=pm)



def positionalencoding(pos):
    """
        Get the positional encoding vector described by Vaswani et al. 2017

        The positional encodings have the same dimension dmodel
        as the embeddings, so that the two can be summed.
    """
    PE = torch.zeros(size=(1, d_model))
    
    for i in range(d_model):
        if i % 2 == 0:
            PE[0][i] = math.sin(i / 10000 ** (2 * (i / d_model)))
        else:
            PE[0][i] = math.cos(i / 10000 ** (2 * (i / d_model)))
    
    return PE


def make_tag_dictionary(datasets : Dataset , tagger : Encoder):
    dict_ = {}

    for dataset in datasets :
        for entry in dataset:
            encoded = tagger.encode(entry)
            for tag in encoded:
                if tag == paradigms.EMPTY_LABEL : break
                if tag in dict_:
                    dict_[tag] += 1
                else :
                    dict_[tag] = 1

    return dict_



tag_dictionary = make_tag_dictionary(datasets = datasets , tagger = tagger)

OUTFEATURES = len(tag_dictionary) # number of classes 
logging.info(f"{OUTFEATURES} many classes are there.")

glove = WordEmbeddings('glove')
sentence = Sentence("This is a sentence .")
glove.embed(sentence)



class Model(nn.Module):
    def __init__(self , in_features , out_features):
        super(Model , self).__init__()
        self.tr = nn.Transformer(in_features , nhead=10 , batch_first=True)
        self.ff = nn.Linear(in_features=in_features , out_features= out_features)
        
    def forward(self , x , tgt):
        tgtnew = self.tr(x,tgt)
        tgtnew = self.ff(tgtnew)
        return tgtnew

        


# tensor = torch.zeros(size=(1,30,100))
# for i , v in enumerate(sentence.tokens): 
#     tensor[0][i] = v.embedding + positionalencoding(i)

# softmax = nn.LogSoftmax(dim = 0)
# tr = Model(in_features = INFEATURES , out_features= OUTFEATURES )
# tgt = torch.rand((1 , 30 , 100))
# loss = F.nll_loss

# for i in range(len(tgt[0])):
#     tgt = tr.forward(tensor , tgt)
#     predictedclass = softmax(tgt[0][i])


