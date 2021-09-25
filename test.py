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
from semantictagger.paradigms import SEQTAG
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

tagger = SEQTAG(
    mult=3 ,
    selectiondelegate=sd,
    reconstruction_module=rm,
    tag_dictionary=tagdictionary,
    rolehandler="complete" ,
    verbshandler="omitverb",
    verbsonly=False, 
    deprel=True
    )

pos_file = "path/to/pos/file"

"""
    UNCOMMENT this to run evaluation.
    then cd evaluation/conll05
    run the script with .tsv files.
"""
def evaluate(debug = False):
    e = eval.EvaluationModule(tagger,dataset_test,ro_file,pd_file, pos_file ,True,True)
    e.createpropsfiles(debug = debug)
    if debug : return 
    with os.popen(f'cd {rootdir}/evaluation/conll05 ; perl srl-eval.pl target.tsv pred.tsv') as output:
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
                break
                



def debugentry(index , spanbased = True):
    i = index 
    entry : CoNLL_U = dataset_test.entries[i]
    encoded = tagger.encode(entry)
    predspans = tagger.spanize(entry.get_words() , encoded=encoded , vlocs = entry.get_vsa() , pos = entry.get_pos())
    targetspans = entry.get_span()
    
    # if spanbased:
    #     predspans = pred.get_span()
    #     targetspans = entry.get_span()
    # else :
    #     predspans = pred.get_depbased()
    #     targetspans = entry.get_depbased()


    dict_ = {"Words" : entry.get_words() , "VSA" : entry.get_vsa() ,  "Encoded" : encoded}
    for j , v in enumerate(targetspans):
        dict_.update({f"Target {j}" :v})

    for j , v in enumerate(predspans):
        dict_.update({f"Pred {j}" :v})

    print(i)
    print(DataFrame(dict_))
    print("\n\n")


collection = DatasetCollection(train=dataset_train,dev=dataset_dev,test=dataset_test)
collection.syntactic_tag_distribution_for_roles()
# collection.dist_xpos_tag_for_predicates()
