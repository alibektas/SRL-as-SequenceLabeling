import pdb 
import os 
import re 
from pathlib import Path
import numpy as np 

from pandas.core.frame import DataFrame
from semantictagger.conllu import CoNLL_U
from semantictagger.dataset import Dataset
from semantictagger.paradigms import DIRECTTAG
from semantictagger.selectiondelegate import SelectionDelegate
import pandas as pd
import eval


rootdir = os.path.dirname(__file__)
sd = SelectionDelegate([lambda x:[x[0]]])


pd_file = Path("./prdtest.tsv")
ro_file = Path("./rotest.tsv")


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

tagger = DIRECTTAG(
    mult=3 ,
    selectiondelegate=sd,
    tag_dictionary=tagdictionary,
    rolehandler="complete" ,
    verbshandler="omitverb",
    verbsonly=False, 
    deprel=True
    )


"""
    UNCOMMENT this to run evaluation.
    then cd evaluation/conll05
    run the script with .tsv files.
"""
e = eval.EvaluationModule(tagger,dataset_test,ro_file,pd_file,True,True)
e.createpropsfiles(debug=True)
# with os.popen(f'cd {rootdir}/evaluation/conll05 ; perl srl-eval.pl target.tsv pred.tsv') as output:
#     while True:
#         line = output.readline()
#         if not line: break
#         line = re.sub(" +" , " " , line)
#         array = line.strip("").strip("\n").split(" ")
#         if len(array) > 2 and array[1] == "Overall": 
#             results = {   
#                 "correct" : np.float(array[2]), 
#                 "excess" : np.float(array[3]),
#                 "missed" : np.float(array[4]),
#                 "recall" : np.float(array[5]),
#                 "precision" : np.float(array[6]),
#                 "f1" : np.float(array[7])
#             }
#             print(results)
#             break
            

        


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth',  None )

def debugentry(index , spanbased = True):
    i = index 
    entry : CoNLL_U = dataset_test.entries[i]
    encoded = tagger.encode(entry)
    pred = tagger.to_conllu(entry.get_words() , encoded=encoded , vlocs = entry.get_vsa())
    
    if spanbased:
        predspans = pred.get_span()
        targetspans = entry.get_span()
    else :
        predspans = pred.get_depbased()
        targetspans = entry.get_depbased()


    dict_ = {"Words" : entry.get_words() , "VSA" : entry.get_vsa() ,  "Encoded" : encoded}
    for j , v in enumerate(targetspans):
        dict_.update({f"Target {j}" :v})

    for j , v in enumerate(predspans):
        dict_.update({f"Pred {j}" :v})

    print(i)
    print(DataFrame(dict_))
    print("\n\n")


# debugentry(6)