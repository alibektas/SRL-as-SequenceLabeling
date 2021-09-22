import os
from re import L
import sys
from pathlib import Path
from flair import embeddings

from flair.embeddings.token import CharacterEmbeddings, FlairEmbeddings, TransformerWordEmbeddings

from semantictagger.conllu import CoNLL_U
from semantictagger.reconstructor import ReconstructionModule
from semantictagger.dataset import Dataset
from semantictagger.paradigms import DIRECTTAG
from semantictagger.selectiondelegate import SelectionDelegate

from flair.data import Corpus , Sentence 
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings , ELMoEmbeddings , OneHotEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from math import ceil , max

import ccformat
import flair,torch

from typing import List

flair.device = torch.device('cuda:1')
curdir = os.path.dirname(__file__)
sys.setrecursionlimit(100000)


def rescuefilesfromModification():
    # Save models from being overwritten.
    modeldir =os.path.join(curdir,'modelout')
    for item in os.listdir(modeldir):
        if os.path.isfile(os.path.join(modeldir, item)):
            saved_file = os.path.join(modeldir , item)
            new_location = os.path.join(modeldir , 'tmp' , item)
            print(f"File Rescue : {saved_file} is being moved to {new_location}")
            os.rename(saved_file , new_location)


def createcolumncorpusfiles():
    train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
    test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
    dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
    dataset_train = Dataset(train_file)
    dataset_test = Dataset(test_file)
    dataset_dev = Dataset(dev_file)
    

    tagdictionary  = dataset_train.tags
    counter = 1 
    for i in tagdictionary:
        tagdictionary[i] = counter
        counter += 1
    tagdictionary["UNK"] = 0

    sd = SelectionDelegate([lambda x: [x[0]]])
    rm = ReconstructionModule()

    tagger = DIRECTTAG(
        mult=3 ,
        selectiondelegate=sd,
        reconstruction_module=rm,
        tag_dictionary=tagdictionary,
        rolehandler="complete" ,
        verbshandler="omitverb",
        verbsonly=False, 
        deprel=True
        )

    ccformat.writecolumncorpus(dataset_train , tagger, filename="train")
    ccformat.writecolumncorpus(dataset_dev , tagger, filename="dev")
    ccformat.writecolumncorpus(dataset_test , tagger, filename="test")



data = ["train.txt" , "test.txt" , "dev.txt"]
for i in range(len(data)) :
    pathtodata = os.path.join(curdir,"data",data[i])

    if os.path.isfile(pathtodata):
        os.remove(pathtodata)

createcolumncorpusfiles() 


# define columns
columns = {0: 'text', 1: 'srl'}

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(os.path.join(curdir,"data"),
                            columns,
                            train_file='train.txt',
                            test_file='test.txt',
                            dev_file='dev.txt')


tag_type = 'srl'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)


def train_lstm(hidden_size : int , lr : float , dropout : float , layer : int , locked_dropout = float , batch_size = int):

    elmo = ELMoEmbeddings("original-all")
    
    postagger = SequenceTagger.load("flair/pos-english")
    postagger.predict(corpus.test, label_name="pos")
    postagger.predict(corpus.train, label_name="pos")
    postagger.predict(corpus.dev, label_name="pos")
    posembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=41)

    frametagger = SequenceTagger.load("bestmodels/predonlymodel.pt")
    frametagger.predict(corpus.dev, label_name="frame")
    frametagger.predict(corpus.test, label_name="frame")
    frametagger.predict(corpus.train, label_name="frame")
    frameembeddings = OneHotEmbeddings(corpus=corpus, field="frame", embedding_length=3)

    StackedEmbeddings(
        embeddings= [elmo , posembeddings , frameembeddings]
    )

    sequencetagger = SequenceTagger(
        hidden_size=hidden_size ,
        embeddings= embeddings ,
        tag_dictionary=tag_dictionary,
        use_crf=False,
        use_rnn= True,
        rnn_layers=1,
        dropout = dropout,
        locked_dropout=locked_dropout
    )

    path = f"model/{lr}-{hidden_size}-{layer}-{dropout}-{locked_dropout}"
    ModelTrainer(sequencetagger,corpus).train(
        base_path= path,
        learning_rate=lr,
        mini_batch_chunk_size=batch_size,
        max_epochs=50
    )
    os.remove(path+"/best-model.pt")
    os.remove(path+"/final-model.pt")



# from hyperopt import hp
# from flair.hyperparameter.param_selection import SearchSpace, Parameter

# # define your search space
# search_space = SearchSpace()
# search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
#     [ELMoEmbeddings("original-all")],
#     [ELMoEmbeddings("original-average")],
#     [ELMoEmbeddings("original-top")],
#     [TransformerWordEmbeddings("bert-large-cased" , use_context=False)],
#     [TransformerWordEmbeddings("bert-base-cased" , use_context=False)],
#     [TransformerWordEmbeddings("roberta-large",use_context=False)],
#     [TransformerWordEmbeddings("roberta-base",use_context=False)],
#     [TransformerWordEmbeddings("roberta-base",use_context=False),]
#     [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward'),ELMoEmbeddings("original-all")]
# ])
# search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
# search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
# search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
# search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.01,0.05, 0.1, 0.2, 0.4])
# search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])


# from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue

# # create the parameter selector
# param_selector = TextClassifierParamSelector(
#     corpus, 
#     False, 
#     'resources/results', 
#     'lstm',
#     max_epochs=50, 
#     training_runs=3,
#     optimization_value=OptimizationValue.DEV_SCORE
# )

# param_selector.optimize(search_space, max_evals=100)