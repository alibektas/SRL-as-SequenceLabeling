from pathlib import Path

from semantictagger.paradigms import DIRECTTAG , RELPOS , MapProtocol , Mapper
from semantictagger.dataset import Dataset

import torch
import flair 

from flair.data import Corpus , Sentence 
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings , ELMoEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
dataset_dev = Dataset(dev_file)

flair.device = torch.device("cuda:1")

# Load POS-tagger.
postagger = SequenceTagger.load("flair/pos-english")


sentence = Sentence(dataset_dev[0].get_sentence())

with torch.no_grad():
    abc = postagger.forward([sentence])

print(type(abc))
print(abc.size() if type(abc) == torch.Tensor else "NoTensor")

