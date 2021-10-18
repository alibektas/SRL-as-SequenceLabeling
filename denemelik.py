import pdb 
import os 
import re 
from pathlib import Path

from numpy.lib.function_base import select
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

path = "model/depless/upos/goldpos/goldframes/3dd9c4df-79d9-4ebc-828a-57fcad88a99c"
path2 = "model/srlreplaced/upos/transformer/1736ff18-0c43-4a92-973f-9dbd665f0000"
path3 = "model/srlextended/upos/transformer/3bffaa33-b233-447a-815b-705fd6e3afba"
path4 = "model/flattened/upos/transformer/86d35810-d243-423f-be67-bdfeb6f355d3"
test_frame_file = Path(f"{path}/data/test_frame.tsv")
test_pos_file = Path(f"{path}/data/test_pos.tsv")
dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
dev_frame_file = Path(f"{path}/data/dev_frame.tsv")
train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
# test_file = Path("./UP_English-EWT/old-test.conllu")
test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
dataset_train = Dataset(train_file)
dataset_test = Dataset(test_file)
dataset_dev = Dataset(dev_file)

dc = dataset_collection.DatasetCollection(dataset_train,dataset_dev,dataset_test)
sd = SelectionDelegate([lambda x: [x[0]]])
rm = ReconstructionModule()

tagger = SRLPOS(
    selectiondelegate=sd,
    reconstruction_module=rm,
    postype=POSTYPE.UPOS,
    frametype=FRAMETYPE.FRAMENUMBER,
    version=RELPOSVERSIONS.DEPLESS
    )




GOLDPREDICATES = True
GOLDPOS = True

    
# ev1 = eval.EvaluationModule(
#         paradigm  = tagger, 
#         dataset = dataset_test,
#         pathroles  = os.path.join(path,"test.tsv"),
#         goldpos = True,
#         goldframes =True ,
#         path_frame_file  = test_frame_file ,
#         path_pos_file  = test_pos_file,
#         mockevaluation = False ,
#         early_stopping = False
#         )

# ev1.reevaluate(path,graph_for_depth=True)
# ev1.role_prediction_by_distance(latex=True)

# tagger2 = SRLPOS(
#     selectiondelegate=sd,
#     reconstruction_module=rm,
#     postype=POSTYPE.UPOS,
#     frametype=FRAMETYPE.FRAMENUMBER,
#     version=RELPOSVERSIONS.SRLREPLACED
#     )


# ev2 = eval.EvaluationModule(
#         paradigm  = tagger2, 
#         dataset = dataset_test,
#         pathroles  = os.path.join(path2,"test.tsv"),
#         goldpos = True,
#         goldframes =True ,
#         path_frame_file  = test_frame_file ,
#         path_pos_file  = test_pos_file,
#         mockevaluation = False ,
#         early_stopping = False
#         )
# ev2.role_prediction_by_distance(latex=True)


path3 = "model/srlextended/upos/goldpos/goldframes/e272465a-13e3-4578-8707-5f46876dda9e"
tagger3 = SRLPOS(
    selectiondelegate=sd,
    reconstruction_module=rm,
    postype=POSTYPE.UPOS,
    frametype=FRAMETYPE.FRAMENUMBER,
    version=RELPOSVERSIONS.SRLEXTENDED
    )

ev3 = eval.EvaluationModule(
        paradigm  = tagger3, 
        dataset = dataset_test,
        pathroles  = os.path.join(path3,"test.tsv"),
        goldpos = True,
        goldframes =True ,
        path_frame_file  = test_frame_file ,
        path_pos_file  = test_pos_file,
        mockevaluation = False ,
        early_stopping = False
        )
ev3.reevaluate(path3)
# ev3.role_prediction_by_distance(latex=True)


# tagger4 = SRLPOS(
#     selectiondelegate=sd,
#     reconstruction_module=rm,
#     postype=POSTYPE.UPOS,
#     frametype=FRAMETYPE.FRAMENUMBER,
#     version=RELPOSVERSIONS.FLATTENED
#     )

# ev4 = eval.EvaluationModule(
#         paradigm  = tagger4, 
#         dataset = dataset_test,
#         pathroles  = os.path.join(path4,"test.tsv"),
#         goldpos = True,
#         goldframes =True ,
#         path_frame_file  = test_frame_file ,
#         path_pos_file  = test_pos_file,
#         mockevaluation = False ,
#         early_stopping = False
#         )
# ev4.role_prediction_by_distance(latex=True)




