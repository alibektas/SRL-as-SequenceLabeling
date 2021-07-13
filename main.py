from modelembed import ModelEmbeddings
import os 
from os import path
from pathlib import Path

from semantictagger.paradigms import DIRECTTAG, PairMapper , RELPOS , MapProtocol , Mapper
from semantictagger.dataset import Dataset

from flair.data import Corpus , Sentence 
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings , ELMoEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.optim import  SGDW
from verbembed import VerbEmbeddings

import pdb 
import ccformat

from posembedding import POSEmbeddings
from modelembed import ModelEmbeddings

import flair,torch

flair.device = torch.device('cuda:1')
curdir = path.dirname(__file__)

pm = PairMapper(
        roles = 
        [
            ("ARG1","V","XARG1V"),
            ("ARG2","V","XARG2V"),
            ("V","ARG1","XVARG1"),
            ("ARG1","ARG0","XARG1-0"),
            #("ARG0","ARG0","XARG0-0"),
        ]
    )


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
    

    #rl = RELPOS()
    #protocol = MapProtocol()
    #mapping = Mapper([dataset_train , dataset_test , dataset_dev], rl , protocol , lowerbound=16)
    #relpos = RELPOS(mapping)

 

    srltagger = DIRECTTAG(2,verbshandler="omitlemma",deprel=True,depreldepth=3 , pairmap=pm)

    ccformat.writecolumncorpus(dataset_train , srltagger, filename="train")
    ccformat.writecolumncorpus(dataset_dev , srltagger, filename="dev")
    ccformat.writecolumncorpus(dataset_test , srltagger, filename="test")




def continuetraining(seqtagger , corpus , start_epoch):
    trainer: ModelTrainer = ModelTrainer(seqtagger , corpus)

    # 7. start training
    trainer.train(path.join(curdir,"modelout"),
                    learning_rate=0.01,
                    mini_batch_size=32,
                    embeddings_storage_mode="gpu",
                    max_epochs=150,
                    write_weights=False)



if not path.isfile(path.join(curdir,"data","train.txt")):
    print("Data not found.")
    createcolumncorpusfiles()
else :
    print("Training data exist.")


print(str(pm))


# define columns
columns = {0: 'text', 1: 'srl'}

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(path.join(curdir,"data"),
                            columns,
                            train_file='train.txt',
                            test_file='test.txt',
                            dev_file='dev.txt')


tag_type = 'srl'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# embedding_types = [
#     ModelEmbeddings()
#   ]

# embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


# tagger: SequenceTagger = SequenceTagger(
#         hidden_size=512,
#         train_initial_hidden_state = True,
#         embeddings=embeddings,
#          tag_dictionary=tag_dictionary,
#          rnn_layers = 1,
#          tag_type=tag_type,
#          dropout=0.3,
#          use_crf=False
#         )

# trainer: ModelTrainer = ModelTrainer(tagger , corpus)

# #7. start training
# trainer.train(path.join(curdir,"modelout"),
#              learning_rate=0.02,
#              mini_batch_size=32,
#              embeddings_storage_mode="gpu",
#              max_epochs=150,
#              write_weights=True)



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
from flair.models import SequenceTagger

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

# 6. initialize trainer with AdamW optimizer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

# 7. run training with XLM parameters (20 epochs, small LR)
from torch.optim.lr_scheduler import OneCycleLR

trainer.train(path.join(curdir,"modelout"),
              learning_rate=5.0e-6,
              mini_batch_size=4,
              mini_batch_chunk_size=1,
              max_epochs=20,
              scheduler=OneCycleLR,
              embeddings_storage_mode='gpu',
              weight_decay=0.,
              )

)
