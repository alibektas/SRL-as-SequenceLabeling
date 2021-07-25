from pathlib import Path
from semantictagger.dataset import Dataset
from semantictagger.paradigms import DIRECTTAG , PairMapper
from flair.data import Sentence

dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
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

srltagger = DIRECTTAG(2,verbshandler="omitlemma",deprel=True,depreldepth=3 , pairmap=pm)

def createcolumncorpusfiles():

    
    train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
    test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
    dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
    dataset_train = Dataset(train_file)
    dataset_test = Dataset(test_file)
    dataset_dev = Dataset(dev_file)
    

    #rl = RELPOS()
    #protocol = MapProtocol()
    #mapping = Mapper([dataset_train , dataset_test , dataset_dev], rl , protocol , lowerbound=16)
    #relpos = RELPOS(mapping)

 

    srltagger = DIRECTTAG(3,verbshandler="omitverb",deprel=True)

    ccformat.writecolumncorpus(dataset_train , srltagger, filename="train")
    ccformat.writecolumncorpus(dataset_dev , srltagger, filename="dev")
    ccformat.writecolumncorpus(dataset_test , srltagger, filename="test")

createcolumncorpusfiles()