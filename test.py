import pdb 
from pathlib import Path
from semantictagger import dataset
from semantictagger.dataset import Dataset
from semantictagger.paradigms import DIRECTTAG , PairMapper
from flair.data import Sentence
import pandas

from typing import Union

pd_file = Path("./prdtest.tsv")
ro_file = Path("./rotest.tsv")


train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
dataset_train = Dataset(train_file)
dataset_test = Dataset(test_file)
dataset_dev = Dataset(dev_file)

import ccformat

pm = PairMapper(
        roles = 
        [
            ("ARG1","V","XARG1V"),
            ("ARG2","V","XARG2V"),
            ("V","ARG2","XARG2V"),
            ("V","ARG1","XARG1V"),
            ("ARG1","ARG0","XARG1-0"),
            #("ARG0","ARG0","XARG0-0"),
        ]
    )

srltagger = DIRECTTAG(3,verbshandler="omitverb",deprel=False)

def getdecodedforentryid(id):
    entry = dataset_dev[id]
    encoded = srltagger.encode(entry)
    verblocs = ["V"  if x != "_" else "" for x in entry.get_vsa()]
    decoded = srltagger.decode(encoded , verblocs)
    words = entry.get_words()

    return pandas.DataFrame(*[zip(*decoded)] , index=words)

def readresults(path : Union[str,Path]):
    entryid = 0
    entry = ["" for d in range(100)]
    counter = 0 

    with path.open() as f:
        while True:
            line = f.readline().replace("\n" , "")
            if line == "" : 
                entryid += 1
                yield entry[:counter]
                entry = ["" for d in range(100)] 
                counter = 0
            else : 
                elems = line.split(" ")
                if len(elems) == 1: 
                    entry[counter] = ""
                elif len(elems) == 2:
                    entry[counter] = ""
                elif len(elems) == 3:
                    entry[counter] = elems[2]
                counter += 1




getpd = readresults(pd_file)
getro = readresults(ro_file)


allcorrect = 0 
somefalse = 0 
correct = 0 
false = 0 
for i in dataset_test.entries:
   
    pdnext = next(getpd)
    ronext = next(getro)
    
    a ,b = srltagger.test((i,pdnext ,ronext))
    correct += a
    false += b
    if b == 0 :
        allcorrect += 1
    else :
        somefalse += 1

acc = correct/(correct+false)

print(f"Sentence level : {allcorrect} , {somefalse} , word-level : {acc}")

