import pdb 
import os 
import re 
from pathlib import Path
from semantictagger import dataset_collection
from semantictagger.dataset_collection import DatasetCollection
import numpy as np 

from pandas.core.frame import DataFrame
from semantictagger import dataset
from semantictagger.conllu import CoNLL_U
from semantictagger.datatypes import FRAMETYPE
from semantictagger.reconstructor import ReconstructionModule
from semantictagger.dataset import Dataset
from semantictagger.paradigms import RELPOSVERSIONS, SRLPOS , POSTYPE 
from semantictagger.selectiondelegate import SelectionDelegate
import pandas as pd
import eval
import ccformat
from math import floor

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth',  None )


# def debugentry(index , spanbased = True):
#     i = index 
#     entry : CoNLL_U = dataset_test.entries[i]
#     encoded = tagger.encode(entry)
#     predconll  = tagger.to_conllu(entry.get_words() , encoded=encoded , vlocs = entry.get_vsa() , pos = entry.get_pos(tagger.postype))
#     predspans = tagger.reconstruct(predconll)


#     targetspans = entry.get_span()

#     dict_ = {"Words" : entry.get_words() , "VSA" : entry.get_vsa() , "POS" : entry.get_pos(tagger.postype), "Encoded" : encoded}
#     for j , v in enumerate(targetspans):
#         dict_.update({f"Target {j}" :v})

#     for j , v in enumerate(predspans):
#         dict_.update({f"Pred {j}" :v})

#     print(i)
#     print(DataFrame(dict_))
#     print("\n\n")


# for i in range(10,15):
#     debugentry(i)

# debugentry(10)



# i : CoNLL_U =None
# for num , i in enumerate(dataset_test.entries):
#     if i.get_sentence().startswith("Bush demoted"):
#         print(num)

# path = "model/flattened/upos/transformer/86d35810-d243-423f-be67-bdfeb6f355d3"
# artificial = Dataset(artifical_entries=[dataset_test[199]])

path  ="model/srlreplaced/upos/goldpos/goldframes/0.75-500-1-0.2-0.3-0-glove-english-1-/vol/fob-vol2/mi16/bektasal/.flair/embeddings/news-forward-0.4.1.pt-2-/vol/fob-vol2/mi16/bektasal/.flair/embeddings/news-backward-0.4.1.pt-3-uposembeddings-4-one-hot-f5624"
test_frame_file = Path(f"{path}/data/test_frame.tsv")
test_pos_file = Path(f"{path}/data/test_pos.tsv")
dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
test_file = Path("./UP_English-EWT/old-test.conllu")
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
    postype=POSTYPE.UPOS,
    frametype=FRAMETYPE.FRAMENUMBER,
    version=RELPOSVERSIONS.SRLREPLACED
    )

GOLDPREDICATES = True
GOLDPOS = True




# ccformat.writecolumncorpus(dataset_train , path , tagger, filename="train" , frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = 1, minfreq = 10)
# ccformat.writecolumncorpus(dataset_dev ,  path, tagger, filename="dev",  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS , downsample = False)
# ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test" ,  frame_gold = GOLDPREDICATES , pos_gold = GOLDPOS ,downsample = False)

# if GOLDPREDICATES:
#     # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_frame",frameonly=True)
#     ccformat.writecolumncorpus(dataset_dev , path , tagger, filename="dev_frame",  frameonly=True)
#     ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test_frame" , frameonly=True)

# if GOLDPOS:
#     # ccformat.writecolumncorpus(dataset_train , tagger, filename="train_pos",posonly=True)
#     ccformat.writecolumncorpus(dataset_dev ,path ,  tagger, filename="dev_pos",  posonly=True)
#     ccformat.writecolumncorpus(dataset_test , path , tagger, filename="test_pos" , posonly=True)
    


ev = eval.EvaluationModule(
        paradigm  = tagger, 
        dataset = dataset_test,
        pathroles  = os.path.join(path,"test.tsv"),
        goldpos = True,
        goldframes =True ,
        path_frame_file  = test_frame_file ,
        path_pos_file  = test_pos_file,
        mockevaluation = False ,
        early_stopping = False
        )

ev.role_prediction_by_distance(latex=True)

# ev.reevaluate(path)


# dict_ = ev.role_prediction_by_distance()
# for i in dict_.items():
#     for j in i[1].items():
#         print(j , end="\t")
#     print()


# a , b , c, d = ev.inspect_learning_behavior(path, 32)

# for i in range(len(a)):
#     print(a[i],b[i],c[i],d[i],a[i]+b[i]+c[i]+d[i])

# print(ev.evaluate("evaluation/conll05/"))

# dc = dataset_collection.DatasetCollection(dataset_train,dataset_dev,dataset_test)
# a , b = dc.semantic_syntactic_head_differences()
# print(a,b)
# print(a/(a+b),b/(a+b))

# abc = dc.pair_mapping_statistics()
# a , b = dc.role_proximity()
# print(a,b , a/(a+b))
# abclist = list(sorted(abc.items(),key=lambda x:x[1]))

# alllabels = 0 
# for i in abclist:
#     alllabels += i[1]
# for i in range(len(abclist)):
#     abclist[i] = (abclist[i][0],abclist[i][1]/alllabels)

# print(abclist)

