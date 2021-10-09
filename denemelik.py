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

tagger = SRLPOS(
        selectiondelegate=sd,
        reconstruction_module=rm,
        tag_dictionary=tagdictionary,
        postype=POSTYPE.UPOS,
        version=RELPOSVERSIONS.SRLREPLACED
        )

pos_file = "path/to/pos/file"






def debugentry(index , spanbased = True):
    i = index 
    entry : CoNLL_U = dataset_test.entries[i]
    encoded = tagger.encode(entry)
    predconll  = tagger.to_conllu(entry.get_words() , encoded=encoded , vlocs = entry.get_vsa() , pos = entry.get_pos(tagger.postype))
    predspans = tagger.reconstruct(predconll)


    targetspans = entry.get_span()

    dict_ = {"Words" : entry.get_words() , "VSA" : entry.get_vsa() , "POS" : entry.get_pos(tagger.postype), "Encoded" : encoded}
    for j , v in enumerate(targetspans):
        dict_.update({f"Target {j}" :v})

    for j , v in enumerate(predspans):
        dict_.update({f"Pred {j}" :v})

    print(i)
    print(DataFrame(dict_))
    print("\n\n")


# for i in range(10,15):
#     debugentry(i)

# debugentry(0)



# i : CoNLL_U =None
# for num , i in enumerate(dataset_test.entries):
#     if i.get_sentence().startswith("Bush demoted"):
#         print(num)

# path = "model/flattened/upos/transformer/86d35810-d243-423f-be67-bdfeb6f355d3"
# artificial = Dataset(artifical_entries=[dataset_test[199]])
ev = eval.EvaluationModule(tagger,dataset=dataset_test,mockevaluation=True)
ev.mockevaluate()

# a , b , c, d = ev.inspect_learning_behavior(path, 32)

# for i in range(len(a)):
#     print(a[i],b[i],c[i],d[i],a[i]+b[i]+c[i]+d[i])

# print(ev.evaluate("evaluation/conll05/"))

# dc = dataset_collection.DatasetCollection(dataset_train,dataset_dev,dataset_test)
# a , b = dc.semantic_syntactic_head_differences()
# print(a,b)
# print(a/(a+b),b/(a+b))