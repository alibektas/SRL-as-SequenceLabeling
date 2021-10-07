from semantictagger import paradigms
from semantictagger.conllu import CoNLL_U
from semantictagger.reconstructor import ReconstructionModule
from semantictagger.dataset import Dataset
from semantictagger.paradigms import RELPOSVERSIONS, SRLPOS , POSTYPE , FRAMETYPE, ParameterError
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
from math import floor, log
import datetime

class EmbeddingWrapper:
    def __init__(self,embedding , embeddingstr):
        self.embedding = embedding
        self.embeddingstr = embeddingstr
    
    def __str__(self) -> str:
        return self.embeddingstr

parser = argparse.ArgumentParser(description='SEQTagging for SRL Training Script.')
parser.add_argument('--DOWNSAMPLE', type = float , help='Downsample ratio [0.1].', default=1.0)
parser.add_argument('--POS-GOLD', type = bool , help='Use GOLD for XPOS/UPOS.' , default=False)
parser.add_argument('--PREDICATE-GOLD', type = bool , help='Use GOLD for predicates.' , default=False)
parser.add_argument('--POS-TYPE', type = str , help='Which type of part of speech tag to use. Options "xpos"/"upos".' , default="upos")
parser.add_argument('--MAX-EPOCH', type = int , help='Number of maximal possible epochs during training.' , default=120)
parser.add_argument('--PARADIGM', type = int , help='Use SRLEXTENDED==1 or SRLREPLACED==2 or FLATTENED==3' , default=1)

dt = datetime.datetime



frametype = FRAMETYPE.FRAMENUMBER
args = parser.parse_args()
GOLDPREDICATES = args.PREDICATE_GOLD
GOLDPOS = args.POS_GOLD
postype = args.POS_TYPE
MAX_EPOCH = args.MAX_EPOCH
PARADIGM = args.PARADIGM
if PARADIGM != 1 and PARADIGM != 2 and PARADIGM!= 3: ParameterError("Parser argument --PARADIGM can either be 1,2 or 3.")

if PARADIGM==1 :
    PARADIGM = RELPOSVERSIONS.SRLEXTENDED 
elif PARADIGM==2:
    PARADIGM = RELPOSVERSIONS.SRLREPLACED 
else:
    PARADIGM= RELPOSVERSIONS.FLATTENED

DOWNSAMPLE = 1
if args.DOWNSAMPLE:
    DOWNSAMPLE = args.DOWNSAMPLE
    if DOWNSAMPLE > 0 and DOWNSAMPLE <= 1:
        print(f"Downsampling is active and set to {DOWNSAMPLE}")
    else :
        Exception(f"Downsampling ratio can't be set to {DOWNSAMPLE}.")

print(f"Max epoch set: {MAX_EPOCH}")


if GOLDPREDICATES is None or GOLDPOS is None:
    Exception("Missing arguments. Use -h option to see what option you should be using.")


if postype is None or postype == "xpos":
    postype : POSTYPE = POSTYPE.XPOS
    print("XPOS is being used.")
else :
    postype : POSTYPE = POSTYPE.UPOS
    print("UPOS is being used.")

if PARADIGM== RELPOSVERSIONS.SRLEXTENDED:
    logfile_name = "srlextended-goldpos-" if GOLDPOS else "srlextended-"
elif PARADIGM == RELPOSVERSIONS.FLATTENED:
    logfile_name = "srlflattened-goldpos-" if GOLDPOS else "srlflattened-"
else:
    logfile_name = "srlreplaced-goldpos-" if GOLDPOS else "srlreplaced-"
    
logfile_name += "goldframes-" if GOLDPREDICATES else ""
logfile_name += "upos" if postype == POSTYPE.UPOS else  "xpos"
tablelogfile_name = logfile_name + "-table.log" 
logfile_name += ".log"

if not os.path.isdir("./logs"): os.mkdir("logs")
path_to_tablelogfile = os.path.join("logs",tablelogfile_name)

mainlogger = logging.getLogger('mainLogger')
mainhandler = logging.FileHandler("./logs/summary.log")
mainlogger.addHandler(mainhandler)
mainlogger.setLevel(logging.DEBUG)

tablelogger = logging.getLogger('tableLogger')
tablehandler = logging.FileHandler(path_to_tablelogfile)
tablelogger.addHandler(tablehandler)
tablelogger.setLevel(logging.DEBUG)


flair.device = torch.device('cuda:1')
curdir = os.path.dirname(__file__)
sys.setrecursionlimit(100000)




def train_lstm(hidden_size : int , lr : float , dropout : float , layer : int , locked_dropout : float , batch_size : int, embeddings : List[EmbeddingWrapper] ,minfreqnumber: int ,  usecrf : bool = False):

    stackedembeddings = StackedEmbeddings(
        embeddings= [x.embedding for x in embeddings]
    )
    
    randid = str(uuid.uuid4())

    max_epoch = MAX_EPOCH
    path = f"model/"
    if PARADIGM == RELPOSVERSIONS.SRLEXTENDED:
        path += "srlextended/"
    elif PARADIGM == RELPOSVERSIONS.SRLREPLACED:
        path += "srlreplaced/"
    elif PARADIGM == RELPOSVERSIONS.FLATTENED:
        path += "flattened/"
    else:
        Exception("Wrong tagger version.")


    path += f"upos/" if postype == POSTYPE.UPOS else "xpos/"
    path += f"goldpos/" if GOLDPOS else "nongoldpos/"
    path += f"goldframes/" if GOLDPREDICATES else "nongoldpredicates/"
    path += f"{randid}"

    if not os.path.isdir(path) : os.makedirs(path)
    os.mkdir(os.path.join(path,"data"))

    test_frame_file = Path(f"{path}/data/test_frame.tsv")
    test_pos_file = Path(f"{path}/data/test_pos.tsv")
    dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
    dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
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
        frametype=frametype,
        version=PARADIGM
        )

 
   
   
    ccformat.writecolumncorpus(dataset_train , path , tagger, filename="train" , frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = floor(DOWNSAMPLE*len(dataset_train)) , minfreq = minfreqnumber)
    ccformat.writecolumncorpus(dataset_dev ,  path, tagger, filename="dev",  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = False)
    ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test" ,  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS ,downsample = False)

    if GOLDPREDICATES:
        # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_frame",frameonly=True)
        ccformat.writecolumncorpus(dataset_dev , path , tagger, filename="dev_frame",  frameonly=True)
        ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test_frame" , frameonly=True)

    if GOLDPOS:
        # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_pos",posonly=True)
        ccformat.writecolumncorpus(dataset_dev ,path ,  tagger, filename="dev_pos",  posonly=True)
        ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test_pos" , posonly=True)
        

    if GOLDPOS and GOLDPREDICATES:
        columns = {0: 'text', 1: 'srl' , 2:'frame' , 3:'pos'}
    elif GOLDPOS:
        columns = {0: 'text', 1: 'srl' , 3:'pos'}
    elif GOLDPREDICATES:
        columns = {0: 'text', 1: 'srl' , 3:'frame'}
    else :
        columns = {0: 'text', 1: 'srl' }


    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(os.path.join(path,"data"),
                                columns,
                                train_file='train.tsv',
                                test_file='test.tsv',
                                dev_file='dev.tsv')

    if not GOLDPOS:
        if tagger.postype == POSTYPE.UPOS:
            upostagger : SequenceTagger = SequenceTagger.load("flair/upos-english-fast")
            upostagger.predict(corpus.test, label_name="pos")
            upostagger.predict(corpus.train, label_name="pos")
            upostagger.predict(corpus.dev, label_name="pos")
            uposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=18)
            upostagger.evaluate(corpus.dev ,out_path = "./data/dev_pos.tsv")
            upostagger.evaluate(corpus.test ,out_path = "./data/test_pos.tsv")
            uposembeddings.name ="upos_emb"

            embeddings.append(EmbeddingWrapper(uposembeddings,"UPOSEmbedding"))


        else:
            xpostagger = SequenceTagger.load("flair/pos-english")
            xpostagger.predict(corpus.test, label_name="pos")
            xpostagger.predict(corpus.train, label_name="pos")
            xpostagger.predict(corpus.dev, label_name="pos")
            xposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=41)
            xpostagger.evaluate(corpus.dev ,out_path = "./data/dev_pos.tsv")
            xpostagger.evaluate(corpus.test ,out_path = "./data/test_pos.tsv")
            xposembeddings.name = "xpos_emb"
            embeddings.append(EmbeddingWrapper(xposembeddings,"XPOSEmbedding"))
    else : 
        if tagger.postype == POSTYPE.XPOS:
            xposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=41)
            xposembeddings.name = "xposembeddings"
            embeddings.append(EmbeddingWrapper(xposembeddings,"XPOSEmbeddingGOLD"))

        else:
            uposembeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=18)
            uposembeddings.name = "uposembeddings"
            embeddings.append(EmbeddingWrapper(uposembeddings,"UPOSEmbeddingGOLD"))





    if not GOLDPREDICATES:
        frametagger = SequenceTagger.load(f"./best_models/predonlymodel.pt")
        frametagger.predict(corpus.dev, label_name="frame")
        frametagger.predict(corpus.test, label_name="frame")
        frametagger.predict(corpus.train, label_name="frame")
        frameembeddings = OneHotEmbeddings(corpus=corpus, field="frame", embedding_length=3)
        frametagger.evaluate(corpus.dev ,out_path = "./data/dev_frame.tsv")
        frametagger.evaluate(corpus.test ,out_path = "./data/test_frame.tsv")
        frameembeddings.name = "frame_emb"
        embeddings.append(EmbeddingWrapper(frameembeddings,"FrameEmbedding"))
    else:
        if frametype == FRAMETYPE.PREDONLY:
            emblen = 3
        elif frametype ==  FRAMETYPE.FRAMENUMBER:
            emblen = 22
        else :
            emblen = 512
        frameembeddings = OneHotEmbeddings(corpus=corpus, field="frame", embedding_length=emblen)
        embeddings.append(EmbeddingWrapper(frameembeddings,"FrameEmbeddingGOLD"))

    tag_type = 'srl'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    sequencetagger = SequenceTagger(
        hidden_size=hidden_size ,
        embeddings= stackedembeddings ,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=usecrf,
        use_rnn= True,
        rnn_layers=layer,
        dropout = dropout,
        locked_dropout=locked_dropout
    )

    
    mainlogger.info(f"{str(dt.now())}\tNEW EXPERIMENT: {path}")

    abc = ModelTrainer(sequencetagger,corpus).train(
        base_path= path,
        learning_rate=lr,
        mini_batch_size = batch_size,
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
    conll05result = "&&&&"
    if results['CoNLL05'] is not None:
        conll05result = f"{results['CoNLL05']['correct']}&{results['CoNLL05']['excess']}&{results['CoNLL05']['missed']}&{results['CoNLL05']['recall']}&{results['CoNLL05']['precision']}&{results['CoNLL05']['f1']}"
    embtext = ""
    start = True
    for i in embeddings:
        if start:
            embtext += f"{str(i)}"
            start = False
        else:
            embtext += f"+{str(i)}"
    if usecrf:
        embtext += f"+CRF"



    tablelogger.info(f"{abc['test_score']}&{embtext}&{lr}&{hidden_size}&{layer}&{dropout}&{locked_dropout}&{batch_size}&{max_epoch}&{DOWNSAMPLE}&{conll05result}&{path}\\\\")

    # data = ["train.tsv" , "test.tsv" , "dev.tsv","train_frame.tsv","test_frame.tsv","dev_frame.tsv","train_pos.tsv","dev_pos.tsv","test_pos.tsv"]
    # for i in range(len(data)):
    #     pathtodata = os.path.join(path,"data",data[i])
    #     if os.path.isfile(pathtodata):
    #         os.remove(pathtodata)
    # os.rmdir(os.path.join(path,"data"))
    os.remove(path+"/best-model.pt")
    os.remove(path+"/final-model.pt")
    

def train(hidden_size,lr,dropout,layer,locked_dropout,batchsize , minfqs):
   
    # flairforward = EmbeddingWrapper(FlairEmbeddings('news-forward'), "FlairNewsForward")
    # flairbackward = EmbeddingWrapper(FlairEmbeddings('news-backward'), "FlairNewsBackward")


    embeddings : List[EmbeddingWrapper] = [
        EmbeddingWrapper(CharacterEmbeddings() , "CharEmbed"),
        EmbeddingWrapper(WordEmbeddings('glove'),"GloVe"),
        EmbeddingWrapper(ELMoEmbeddings('small-top'),"ELMoSmallTop")
        # flairforward,
        # flairbackward
    ]
    
    
    for h in hidden_size:
        for j in lr:
            for k in dropout:
                for l in layer:
                    for m in locked_dropout:
                        for n in batchsize:
                            for minfq in minfqs:
                                train_lstm(hidden_size = h , lr = j , dropout =k , layer = l , locked_dropout = m , batch_size=n,embeddings=embeddings, minfreqnumber=minfq,usecrf=False) 


def traintransformer():

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    from flair.embeddings import TransformerWordEmbeddings


    # embeddings = [EmbeddingWrapper(TransformerWordEmbeddings(
    #     model='roberta-large',
    #     layers="-1",
    #     subtoken_pooling="first",
    #     fine_tune=True,
    #     use_context=True
    # ),"RobertaLargeFineTune")]

    embeddings = [EmbeddingWrapper(TransformerWordEmbeddings(
        model='bert-base-cased',
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True
    ),"BertBaseCasedFineTune")]


    randid = str(uuid.uuid4())

    max_epoch = MAX_EPOCH
    path = f"model/"
    if PARADIGM == RELPOSVERSIONS.SRLEXTENDED:
        path += "srlextended/"
    elif PARADIGM == RELPOSVERSIONS.SRLREPLACED:
        path += "srlreplaced/"
    elif PARADIGM == RELPOSVERSIONS.FLATTENED:
        path += "flattened/"
    else:
        Exception("Wrong tagger version.")


    path += f"upos/" if postype == POSTYPE.UPOS else "xpos/"
    path += f"transformer/"
    path += f"{randid}"

    if not os.path.isdir(path) : os.makedirs(path)
    os.mkdir(os.path.join(path,"data"))

    test_frame_file = Path(f"{path}/data/test_frame.tsv")
    test_pos_file = Path(f"{path}/data/test_pos.tsv")
    dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
    dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
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
        frametype=frametype,
        version=PARADIGM
        )

 
   
   
    ccformat.writecolumncorpus(dataset_train , path , tagger, filename="train" , frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = floor(DOWNSAMPLE*len(dataset_train)) , minfreq = 10)
    ccformat.writecolumncorpus(dataset_dev ,  path, tagger, filename="dev",  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = False)
    ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test" ,  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS ,downsample = False)


    if GOLDPREDICATES:
        # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_frame",frameonly=True)
        ccformat.writecolumncorpus(dataset_dev , path , tagger, filename="dev_frame",  frameonly=True)
        ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test_frame" , frameonly=True)

    if GOLDPOS:
        # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_pos",posonly=True)
        ccformat.writecolumncorpus(dataset_dev ,path ,  tagger, filename="dev_pos",  posonly=True)
        ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test_pos" , posonly=True)
        
    
    columns = {0: 'text', 1: 'srl' }


    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(os.path.join(path,"data"),
                                columns,
                                train_file='train.tsv',
                                test_file='test.tsv',
                                dev_file='dev.tsv')

    
    tag_type = 'srl'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    seqtagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings[0].embedding,
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
                max_epochs=max_epoch,
                scheduler=OneCycleLR,
                embeddings_storage_mode='gpu',
                weight_decay=0.,
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
    conll05result = "&&&&"
    if results['CoNLL05'] is not None:
        conll05result = f"{results['CoNLL05']['correct']}&{results['CoNLL05']['excess']}&{results['CoNLL05']['missed']}&{results['CoNLL05']['recall']}&{results['CoNLL05']['precision']}&{results['CoNLL05']['f1']}"
    embtext = ""
    start = True
    for i in embeddings:
        if start:
            embtext += f"{str(i)}"
            start = False
        else:
            embtext += f"+{str(i)}"



    tablelogger.info(f"{abc['test_score']}&{embtext}&{lr}&{hidden_size}&{layer}&{dropout}&{locked_dropout}&{batch_size}&{max_epoch}&{DOWNSAMPLE}&{conll05result}&{path}\\\\")

    # data = ["train.tsv" , "test.tsv" , "dev.tsv","train_frame.tsv","test_frame.tsv","dev_frame.tsv","train_pos.tsv","dev_pos.tsv","test_pos.tsv"]
    # for i in range(len(data)):
    #     pathtodata = os.path.join(path,"data",data[i])
    #     if os.path.isfile(pathtodata):
    #         os.remove(pathtodata)
    # os.rmdir(os.path.join(path,"data"))
    os.remove(path+"/best-model.pt")
    os.remove(path+"/final-model.pt")


lr = [0.95]
hidden_size = [800]
layer =[2]
dropout=[0.2]
locked_dropout = [0]
batchsize=[32]
minfqs=[10]
train(hidden_size,lr,dropout,layer,locked_dropout,batchsize,minfqs)
# traintransformer()
