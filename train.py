from semantictagger.conllu import CoNLL_U
from semantictagger.reconstructor import ReconstructionModule
from semantictagger.dataset import Dataset
from semantictagger.paradigms import SRLPOS , POSTYPE , FRAMETYPE
from semantictagger.selectiondelegate import SelectionDelegate

from flair.embeddings.token import CharacterEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, WordEmbeddings
from flair.data import Corpus 
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings , ELMoEmbeddings , OneHotEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import ccformat
import flair,torch
from typing import List
import logging
import uuid 
import logging
import eval 
import argparse
import numpy as np 
import re
import os
import sys
from pathlib import Path
import pdb
from math import floor

parser = argparse.ArgumentParser(description='SEQTagging for SRL Training Script.')
parser.add_argument('--DOWNSAMPLE', type = float , help='Downsample ratio [0.1].', default=1.0)
parser.add_argument('--POS-GOLD', type = bool , help='Use GOLD for XPOS/UPOS.' , default=False)
parser.add_argument('--PREDICATE-GOLD', type = bool , help='Use GOLD for predicates.' , default=False)
parser.add_argument('--POS-TYPE', type = str , help='Which type of part of speech tag to use. Options "xpos"/"upos".' , default="upos")
parser.add_argument('--MAX-EPOCH', type = int , help='Number of maximal possible epochs during training.' , default=120)

frametype = FRAMETYPE.FRAMENUMBER
args = parser.parse_args()
GOLDPREDICATES = args.PREDICATE_GOLD
GOLDPOS = args.POS_GOLD
postype = args.POS_TYPE
MAX_EPOCH = args.MAX_EPOCH
DOWNSAMPLE = 1

# GOLDPREDICATES = True
# GOLDPOS = True
# postype = "upos"
# MAX_EPOCH = 1
# DOWNSAMPLE = 0.2


if args.DOWNSAMPLE:
    DOWNSAMPLE = args.DOWNSAMPLE
    if DOWNSAMPLE > 0 and DOWNSAMPLE <= 1:
        print(f"Downsampling is active and set to {DOWNSAMPLE}")
    else :
        Exception(f"Downsampling ratio can't be set to {DOWNSAMPLE}.")

print(f"Max epoch set: {MAX_EPOCH}")

# GOLDPREDICATES = True
# GOLDPOS = True
# postype = POSTYPE.XPOS
# DOWNSAMPLE = 0.2

if GOLDPREDICATES is None or GOLDPOS is None:
    Exception("Missing arguments. Use -h option to see what option you should be using.")


if postype is None or postype == "xpos":
    postype : POSTYPE = POSTYPE.XPOS
    print("XPOS is being used.")
else :
    postype : POSTYPE = POSTYPE.UPOS
    print("UPOS is being used.")

logfile_name = "goldpos-" if GOLDPOS else ""
logfile_name += "goldframes-" if GOLDPREDICATES else ""
logfile_name += "upos" if postype == POSTYPE.UPOS else  "xpos"
logfile_name += ".log"
if not os.path.isdir("./logs"): os.mkdir("logs")
path_to_logfile = os.path.join("logs",logfile_name)

logger = logging.getLogger('res')
handler = logging.FileHandler(path_to_logfile)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


mainlogger = logging.getLogger('mainLogger')
mainhandler = logging.FileHandler("./logs/summary.log")
mainlogger.addHandler(mainhandler)
mainlogger.setLevel(logging.DEBUG)

# flair.device = torch.device('cuda:1')
curdir = os.path.dirname(__file__)
sys.setrecursionlimit(100000)

test_frame_file = Path("./data/test_frame.tsv")
test_pos_file = Path("./data/test_pos.tsv")
dev_frame_file = Path("./data/dev_frame.tsv")
dev_frame_file = Path("./data/dev_frame.tsv")
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


tagger = SRLPOS(
        selectiondelegate=sd,
        reconstruction_module=rm,
        tag_dictionary=tagdictionary,
        postype=postype,
        frametype=frametype
        )



data = ["train.tsv" , "test.tsv" , "dev.tsv","train_frame.tsv","test_frame.tsv","dev_frame.tsv","train_pos.tsv","dev_pos.tsv","test_pos.tsv"]
for i in range(len(data)) :
    pathtodata = os.path.join(curdir,"data",data[i])
    if os.path.isfile(pathtodata):
        os.remove(pathtodata)

sizes = [floor(DOWNSAMPLE*len(dataset_train)),floor(DOWNSAMPLE*len(dataset_dev)) ,floor(DOWNSAMPLE*len(dataset_test))]
ccformat.writecolumncorpus(dataset_train , tagger, filename="train" , frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = sizes[0])
ccformat.writecolumncorpus(dataset_dev , tagger, filename="dev",  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = False)
ccformat.writecolumncorpus(dataset_test , tagger, filename="test" ,  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS ,downsample = False)

if GOLDPREDICATES:
    # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_frame",frameonly=True)
    ccformat.writecolumncorpus(dataset_dev , tagger, filename="dev_frame",  frameonly=True)
    ccformat.writecolumncorpus(dataset_test , tagger, filename="test_frame" , frameonly=True)

if GOLDPOS:
    # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_pos",posonly=True)
    ccformat.writecolumncorpus(dataset_dev , tagger, filename="dev_pos",  posonly=True)
    ccformat.writecolumncorpus(dataset_test , tagger, filename="test_pos" , posonly=True)
    

if GOLDPOS and GOLDPREDICATES:
    columns = {0: 'text', 1: 'srl' , 2:'frame' , 3:'pos'}
elif GOLDPOS:
    columns = {0: 'text', 1: 'srl' , 3:'pos'}
elif GOLDPREDICATES:
    columns = {0: 'text', 1: 'srl' , 3:'frame'}
else :
    columns = {0: 'text', 1: 'srl' }


# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(os.path.join(curdir,"data"),
                            columns,
                            train_file='train.tsv',
                            test_file='test.tsv',
                            dev_file='dev.tsv')


tag_type = 'srl'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)


def train_lstm(hidden_size : int , lr : float , dropout : float , layer : int , locked_dropout : float , batch_size : int,embeddings):

    stackedembeddings = StackedEmbeddings(
        embeddings= embeddings
    )

    randid = str(uuid.uuid1())[0:5]
    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")
    

    sequencetagger = SequenceTagger(
        hidden_size=hidden_size ,
        embeddings= stackedembeddings ,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=False,
        use_rnn= True,
        rnn_layers=layer,
        dropout = dropout,
        locked_dropout=locked_dropout
    )

    max_epoch = MAX_EPOCH
    path = f"model/"
    path += f"upos/" if tagger.postype == POSTYPE.UPOS else "xpos/"
    path += f"goldpos/" if GOLDPOS else "nongoldpos/"
    path += f"goldframes/" if GOLDPREDICATES else "nongoldpredicates/"
    if not os.path.isdir(path) : os.makedirs(path)
    path += f"{lr}-{hidden_size}-{layer}-{dropout}-{locked_dropout}"
    for l in embeddings :
        path += f"-{str(l.name)}"
    path += f"-{randid}"
    mainlogger.info(f"NEW EXPERIMENT: {path}")
    logger.info(f"EXPERIMENT : {path}")
    logger.info(f"\tlr:{lr}")
    logger.info(f"\thidden size:{hidden_size}")
    logger.info(f"\tlayer:{layer}")
    logger.info(f"\tdropout:{dropout}")
    logger.info(f"\tlocked dropout:{locked_dropout}")
    logger.info(f"\tbatch size:{batch_size}")
    logger.info(f"\tmax epoch:{max_epoch}")
    logger.info(f"\tdownsample ratio:{DOWNSAMPLE}")


    abc = ModelTrainer(sequencetagger,corpus).train(
        base_path= path,
        learning_rate=lr,
        mini_batch_chunk_size=batch_size,
        max_epochs=max_epoch,
        embeddings_storage_mode="gpu"
    )

         
   
    e = eval.EvaluationModule(
        paradigm  = tagger, 
        dataset = dataset_test,
        pathroles  = os.path.join(path,"test.tsv"),
        goldpos = GOLDPOS,
        goldframes = GOLDPREDICATES,
        path_frame_file  = test_frame_file ,
        path_pos_file  = test_pos_file,
        mockevaluation = False ,
        early_stopping = False
        )

    results = e.evaluate(path = path)

    logger.info(f"F1 Score : \t{abc['test_score']}")
    for k in results.items():
        name = k[0]
        content = k[1]
        if content is None:
            logger.info(f"{name} tests failed")
            continue

        if e.goldpos and e.goldframes:
            logger.info(f"{name} GOLD FRAME AND GOLD POS")
        elif e.goldpos:
            logger.info(f"{name} GOLD POS")
        elif e.goldframes:
            logger.info(f"{name} GOLD FRAME")
        for i in k[1].items():
            logger.info(f"\t{i[0]}\t{i[1]}")

    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")
    
    os.remove(path+"/best-model.pt")
    os.remove(path+"/final-model.pt")
    

def train(hidden_size,lr,dropout,layer,locked_dropout,batchsize):
   

    glove = WordEmbeddings('glove')
    glove.name = "glove-english"
    embeddings = [glove]
   
    # elmo = ELMoEmbeddings("small-all")
    # elmo.name = "elmo-small-all"
    # embeddings = [elmo]
    
    if not GOLDPOS:
        if tagger.postype == POSTYPE.UPOS:
            upostagger : SequenceTagger = SequenceTagger.load("flair/upos-english-fast")
            upostagger.predict(corpus.test, label_name="pos")
            upostagger.predict(corpus.train, label_name="pos")
            upostagger.predict(corpus.dev, label_name="pos")
            uposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=17)
            upostagger.evaluate(corpus.dev ,out_path = "./data/dev_pos.tsv")
            upostagger.evaluate(corpus.test ,out_path = "./data/test_pos.tsv")
            uposembeddings.name ="upos_emb"

            embeddings.append(uposembeddings)


        else:
            xpostagger = SequenceTagger.load("flair/pos-english")
            xpostagger.predict(corpus.test, label_name="pos")
            xpostagger.predict(corpus.train, label_name="pos")
            xpostagger.predict(corpus.dev, label_name="pos")
            xposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=41)
            xpostagger.evaluate(corpus.dev ,out_path = "./data/dev_pos.tsv")
            xpostagger.evaluate(corpus.test ,out_path = "./data/test_pos.tsv")
            xposembeddings.name = "xpos_emb"
            embeddings.append(xposembeddings)
    else : 
        if tagger.postype == POSTYPE.XPOS:
            xposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=17)
        else:
            xposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=41)




    if not GOLDPREDICATES:
        frametagger = SequenceTagger.load(f"./best_models/predonlymodel.pt")
        frametagger.predict(corpus.dev, label_name="frame")
        frametagger.predict(corpus.test, label_name="frame")
        frametagger.predict(corpus.train, label_name="frame")
        frameembeddings = OneHotEmbeddings(corpus=corpus, field="frame", embedding_length=3)
        frametagger.evaluate(corpus.dev ,out_path = "./data/dev_frame.tsv")
        frametagger.evaluate(corpus.test ,out_path = "./data/test_frame.tsv")
        frameembeddings.name = "frame_emb"
        embeddings.append(frameembeddings)
    else:
        if frametype == FRAMETYPE.PREDONLY:
            emblen = 3
        elif frametype ==  FRAMETYPE.FRAMENUMBER:
            emblen = 22
        else :
            emblen = 512
        frameembeddings = OneHotEmbeddings(corpus=corpus, field="frame", embedding_length=emblen)


   
    
    for h in hidden_size:
        for j in lr:
            for k in dropout:
                for l in layer:
                    for m in locked_dropout:
                        for n in batchsize:
                            train_lstm(hidden_size = h , lr = j , dropout =k , layer = l , locked_dropout = m , batch_size=n,embeddings=embeddings)


def traintransformer():

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    from flair.embeddings import TransformerWordEmbeddings

    embeddings = TransformerWordEmbeddings(
        model='bert-large-uncased',
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True
    )


    randid = str(uuid.uuid1())[0:5]
    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")
    path = f"model/"
    path += f"upos/" if tagger.postype == POSTYPE.UPOS else "xpos/"
    path += f"goldpos/" if GOLDPOS else "nongoldpos/"
    path += f"goldframes/" if GOLDPREDICATES else "nongoldpredicates/"
    if not os.path.isdir(path) : os.makedirs(path)
    path += f"{embeddings.name}"
    path += f"-{randid}"
    logger.info(f"EXPERIMENT : {path}")
    logger.info("Tranformer")
    logger.info(f"\t downsample ratio:{DOWNSAMPLE}")


    seqtagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type='srl',
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )


    # 6. initialize trainer with AdamW optimizer
    from flair.trainers import ModelTrainer

    trainer = ModelTrainer(seqtagger, corpus, optimizer=torch.optim.AdamW)

    # 7. run training with XLM parameters (20 epochs, small LR)
    from torch.optim.lr_scheduler import OneCycleLR
    abc = trainer.train(
                base_path=path,
                learning_rate=5.0e-6,
                mini_batch_size=4,
                mini_batch_chunk_size=1,
                max_epochs=20,
                scheduler=OneCycleLR,
                embeddings_storage_mode='gpu',
                weight_decay=0.,
                )

   

    logger.info(embeddings.name)
   
    e = eval.EvaluationModule(
        paradigm  = tagger, 
        dataset = dataset_test,
        pathroles  = os.path.join(path,"test.tsv"),
        goldpos = GOLDPOS,
        early_stopping = len(corpus.test),
        goldframes = GOLDPREDICATES,
        path_frame_file  = test_frame_file ,
        path_pos_file  = test_pos_file,
        mockevaluation = False ,
        )

    e.createpropsfiles(saveloc = path , debug = False)
    results = e.evaluate(path)
    logger.info(f"F1 Score : \t{abc['test_score']}")

    for formats in ["conll05"]:
        if formats in results:
            if results[formats] is None:
                logger.info(f"{formats} tests failed")
            if e.goldpos and e.goldframes:
                logger.info(f"{formats} GOLD FRAME AND GOLD POS")
            elif e.goldpos:
                logger.info(f"{formats} GOLD POS")
            elif e.goldframes:
                logger.info(f"{formats} GOLD FRAME")
            for i in formats.items():
                logger.info(f"\t{i[0]}\t{i[1]}")
        
    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")
    
    os.remove(path+"/best-model.pt")
    os.remove(path+"/final-model.pt")



lr = [0.2]
hidden_size = [1]
layer =[1]
dropout=[0.2]
locked_dropout = [0.1]
batchsize=[16]
train(hidden_size,lr,dropout,layer,locked_dropout,batchsize)
# traintransformer()
