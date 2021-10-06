import pdb 
import os 
import re 
from pathlib import Path
from semantictagger.dataset_collection import DatasetCollection
import numpy as np 

from pandas.core.frame import DataFrame
from semantictagger import dataset
from semantictagger.conllu import CoNLL_U
from semantictagger.reconstructor import ReconstructionModule
from semantictagger.dataset import Dataset
from semantictagger.paradigms import RELPOSVERSIONS, SRLPOS , POSTYPE 
from semantictagger.selectiondelegate import SelectionDelegate
import pandas as pd
import eval


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth',  None )


rootdir = os.path.dirname(__file__)
sd = SelectionDelegate([lambda x: [x[0]]])
rm = ReconstructionModule()
pd_file = Path("./test/prdtest.tsv")
ro_file = Path("./test/rotest.tsv")


train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
dataset_train = Dataset(train_file)
dataset_test = Dataset(test_file)

dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
dataset_dev = Dataset(dev_file)


tagdictionary  = dataset_train.tags
counter = 1 
for i in tagdictionary:
    tagdictionary[i] = counter
    counter += 1
tagdictionary["UNK"] = 0

# tagger = SRLPOS(
#         selectiondelegate=sd,
#         reconstruction_module=rm,
#         tag_dictionary=tagdictionary,
#         postype=POSTYPE.UPOS
#         )

pos_file = "path/to/pos/file"





def debugentry(index , tagger ,  spanbased = True):
    i = index 
    entry : CoNLL_U = dataset_test.entries[i]
    encoded = tagger.encode(entry)
    # predspans = tagger.spanize(entry.get_words() , encoded=encoded , vlocs = entry.get_vsa() , pos = entry.get_by_tag("xpos"))
    # targetspans = entry.get_span()

    # dict_ = {"Words" : entry.get_words() , "VSA" : entry.get_vsa() ,  "Encoded" : encoded}
    if tagger.version == RELPOSVERSIONS.SRLEXTENDED or tagger.version == RELPOSVERSIONS.SRLREPLACED:
        dict_ = {"Words" : entry.get_words() , "VSA" : entry.get_vsa() , "POS":entry.get_pos(POSTYPE.UPOS)  ,"Encoded" : encoded}
    else :
        dict_ = {"Words" : entry.get_words() , "POS":entry.get_pos(POSTYPE.UPOS)  ,"Encoded" : encoded}

    # for j , v in enumerate(targetspans):
    #     dict_.update({f"Target {j}" :v})

    # for j , v in enumerate(predspans):
    #     dict_.update({f"Pred {j}" :v})

    print(i)
    print(DataFrame(dict_).to_latex())
    print("\n\n")

a = RELPOSVERSIONS.ORIGINAL
b = RELPOSVERSIONS.FLATTENED
c = RELPOSVERSIONS.SRLEXTENDED
d = RELPOSVERSIONS.SRLREPLACED



for i in [a,b,c,d]:
    tagger = SRLPOS(
            selectiondelegate=sd,
            reconstruction_module=rm,
            tag_dictionary=tagdictionary,
            version=i,
            postype=POSTYPE.UPOS
            )
    debugentry(0 , tagger)