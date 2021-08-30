import os
import sys
from pathlib import Path

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


def trainlstm( lrs : List[float] , drops : List[float] , hsizes : List[int] , lsizes : List[int]):
    
    frametagger = SequenceTagger.load("./best_models/predonlymodel.pt")
    frametagger.predict(corpus.train, label_name="predicted_frame")
    frameembedding = OneHotEmbeddings(corpus=corpus, field="predicted_frame", embedding_length=1)

    embedding_types = [
        ELMoEmbeddings("original-all"), 
        frameembedding
    ]
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


    for lr in lrs:
        for drop in drops:
            for hsize in hsizes:
                for lsize in lsizes:
                    tagger: SequenceTagger = SequenceTagger(
                    hidden_size=hsize,
                    train_initial_hidden_state = False,
                    embeddings=embeddings,
                    tag_dictionary=tag_dictionary,
                    rnn_layers = 1,
                    tag_type=tag_type,
                    dropout=drop,
                    use_crf=False
                    )
                    
                    trainer: ModelTrainer = ModelTrainer(tagger , corpus)
                    path = os.path.join(curdir,"model",f"{hsize}-{lsize}-{drop}-{lr}")
                    os.mkdir(path)
                    
                    #7. start training
                    trainer.train(
                                path,
                                learning_rate=lr,
                                mini_batch_size=32,
                                embeddings_storage_mode="gpu",
                                max_epochs=80,
                                write_weights=False
                            )






def traintransformer():

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    from flair.embeddings import TransformerWordEmbeddings

    embeddings = TransformerWordEmbeddings(
        model='xlm-roberta-large',
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True,
    )

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    #from flair.models import SequenceTagger

    #tagger = SequenceTagger.load(path.join(curdir,"modelout","bertabest","final-model.pt"))

    # 6. initialize trainer with AdamW optimizer
    from flair.trainers import ModelTrainer

    trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

    # # 7. run training with XLM parameters (20 epochs, small LR)
    from torch.optim.lr_scheduler import OneCycleLR
    continuetraining(tagger,corpus,20)
    trainer.train(path.join(curdir,"modelout"),
                learning_rate=5.0e-6,
                mini_batch_size=4,
                mini_batch_chunk_size=1,
                max_epochs=20,
                scheduler=OneCycleLR,
                embeddings_storage_mode='gpu',
                weight_decay=0.,
                )


trainlstm(
    lrs =[0.01,0.001],
    drops = [0,0.2,0.4],
    hsizes = [300,512],
    lsizes = [1]
    )
