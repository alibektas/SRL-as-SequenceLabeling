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


parser = argparse.ArgumentParser(description='SEQTagging for SRL Training Script.')
parser.add_argument('--POS-GOLD', type = bool , help='Use GOLD for XPOS/UPOS.')
parser.add_argument('--PREDICATE-GOLD', type = bool , help='Use GOLD for predicates.')
parser.add_argument('--POS-TYPE', type = str , help='Which type of part of speech tag to use. Options "xpos"/"upos".')

args = parser.parse_args()
GOLDPREDICATES = args.PREDICATE_GOLD
GOLDPOS = args.POS_GOLD
postype = args.POS_TYPE


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
if not os.path.isdir("./logs"): os.mkdir("logs")
path_to_logfile = os.path.join("logs",logfile_name)

logger = logging.getLogger('res')
handler = logging.FileHandler(path_to_logfile)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

flair.device = torch.device('cuda:1')
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
        postype=postype
        )



data = ["train.tsv" , "test.tsv" , "dev.tsv","train_frame.tsv","test_frame.tsv","dev_frame.tsv","train_pos.tsv","dev_pos.tsv","test_pos.tsv"]
for i in range(len(data)) :
    pathtodata = os.path.join(curdir,"data",data[i])
    if os.path.isfile(pathtodata):
        os.remove(pathtodata)


ccformat.writecolumncorpus(dataset_train , tagger, filename="train" , frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS)
ccformat.writecolumncorpus(dataset_dev , tagger, filename="dev",  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS)
ccformat.writecolumncorpus(dataset_test , tagger, filename="test" ,  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS)

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
corpus = corpus.downsample(0.1)


tag_type = 'srl'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)


def train_lstm(hidden_size : int , lr : float , dropout : float , layer : int , locked_dropout : float , batch_size : int,embeddings):

    
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

    max_epoch = 1
    path = f"model/"
    path += f"upos/" if tagger.postype == POSTYPE.UPOS else "xpos/"
    path += f"goldpos/" if GOLDPOS else "nongoldpos/"
    path += f"goldframes/" if GOLDPREDICATES else "nongoldpredicates/"
    if not os.path.isdir(path) : os.makedirs(path)
    path += f"{lr}-{hidden_size}-{layer}-{dropout}-{locked_dropout}"
    for l in embeddings :
        path += f"-{str(l)}"
    path += f"-{randid}"
    logger.info(f"EXPERIMENT : {path}")
    logger.info(f"\tlr:{lr}")
    logger.info(f"\thidden size:{hidden_size}")
    logger.info(f"\tlayer:{layer}")
    logger.info(f"\tdropout:{dropout}")
    logger.info(f"\tlocked dropout:{locked_dropout}")
    logger.info(f"\tbatch size:{batch_size}")
    logger.info(f"\tmax epoch:{max_epoch}")


    logger.info(str(stackedembeddings))

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
        early_stopping = len(corpus.test),
        pathroles  = os.path.join(path,"test.tsv"),
        goldpos = GOLDPOS,
        goldframes = GOLDPREDICATES,
        path_frame_file  = test_frame_file ,
        path_pos_file  = test_pos_file,
        span_based = True,
        mockevaluation = False ,
        )

    conll05terminates = False
    e.createpropsfiles(saveloc = path , debug = False)
    with os.popen(f'perl ./evaluation/conll05/srl-eval.pl {path}/target-props.tsv {path}/predicted-props.tsv') as output:
        while True:
            line = output.readline()
            if not line: break
            line = re.sub(" +" , " " , line)
            array = line.strip("").strip("\n").split(" ")
            if len(array) > 2 and array[1] == "Overall": 
                results = {   
                    "correct" : np.float(array[2]), 
                    "excess" : np.float(array[3]),
                    "missed" : np.float(array[4]),
                    "recall" : np.float(array[5]),
                    "precision" : np.float(array[6]),
                    "f1" : np.float(array[7])
                }
                conll05terminates = True
                break
        logger.info(f"F1 Score : \t{abc['test_score']}")
        if conll05terminates:
            if e.goldpos and e.goldframes:
                logger.info("CoNLL 05 GOLD FRAME AND GOLD POS")
            elif e.goldpos:
                logger.info("CoNLL 05 GOLD POS")
            elif e.goldframes:
                logger.info("CoNLL 05 GOLD FRAME")
            for i in results.items():
                logger.info(f"\t{i[0]}\t{i[1]}")
        else:
            logger.info("CoNLL05 tests failed")

    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")
    
    os.remove(path+"/best-model.pt")
    os.remove(path+"/final-model.pt")
    

def train(hidden_size,lr,dropout,layer,locked_dropout,batchsize):
   
    glove = WordEmbeddings('glove')
    glove.name = "glove-english"
    embeddings = [glove]
    # elmo = ELMoEmbeddings("small-top")
    # elmo.name = "elmo-small-top"
    # embeddings = [elmo]
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
        goldframes = GOLDPREDICATES,
        path_frame_file  = test_frame_file ,
        path_pos_file  = test_pos_file,
        span_based = True,
        mockevaluation = False ,
        )

    conll05terminates = False
    e.createpropsfiles(saveloc = path , debug = False)
    with os.popen(f'perl ./evaluation/conll05/srl-eval.pl {path}/target-props.tsv {path}/predicted-props.tsv') as output:
        while True:
            line = output.readline()
            if not line: break
            line = re.sub(" +" , " " , line)
            array = line.strip("").strip("\n").split(" ")
            if len(array) > 2 and array[1] == "Overall": 
                results = {   
                    "correct" : np.float(array[2]), 
                    "excess" : np.float(array[3]),
                    "missed" : np.float(array[4]),
                    "recall" : np.float(array[5]),
                    "precision" : np.float(array[6]),
                    "f1" : np.float(array[7])
                }
                conll05terminates = True
                break
        logger.info(f"F1 Score : \t{abc['test_score']}")
        if conll05terminates:
            if e.goldpos and e.goldframes:
                logger.info("CoNLL 05 GOLD FRAME AND GOLD POS")
            elif e.goldpos:
                logger.info("CoNLL 05 GOLD POS")
            elif e.goldframes:
                logger.info("CoNLL 05 GOLD FRAME")
            for i in results.items():
                logger.info(f"\t{i[0]}\t{i[1]}")
        else:
            logger.info("CoNLL05 tests failed")

    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++")
    
    os.remove(path+"/best-model.pt")
    os.remove(path+"/final-model.pt")



lr = [0.4]
hidden_size = [1]
layer =[1]
dropout=[0.2]
locked_dropout = [0.1]
batchsize=[16]
train(hidden_size,lr,dropout,layer,locked_dropout,batchsize)
# traintransformer()
