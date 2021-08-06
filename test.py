import pdb 
from pathlib import Path

from pandas.core.frame import DataFrame
from semantictagger import conllu, dataset
from semantictagger.dataset import Dataset
from semantictagger.paradigms import DIRECTTAG , PairMapper
from flair.data import Sentence, Span
import pandas
import eval

from typing import Union

pd_file = Path("./prdtest.tsv")
ro_file = Path("./rotest.tsv")


train_file = Path("./UP_English-EWT/en_ewt-up-train.conllu")
test_file = Path("./UP_English-EWT/en_ewt-up-test.conllu")
dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
dataset_train = Dataset(train_file)
dataset_test = Dataset(test_file)
dataset_dev = Dataset(dev_file)

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

srltagger = DIRECTTAG(5,verbshandler="omitverb",deprel=True)

# correct = 0 
# false = 0 
# for i  , v in enumerate(dataset_test):
#     a ,b  = srltagger.test(v)
    
#     correct += a 
#     false += b 
    
# print(f"{correct / (correct+false)}")
evl = eval.EvaluationModule(srltagger , dataset_test , ro_file , pd_file , mockevaluation=True)
# a = evl.single(verbose = True)
# print(a)
# corr.  excess  missed    prec.    rec.      F1
# 5910    2629    3918    69.21   60.13   64.35
# 6297    2509    3531    71.51   64.07   67.59
# 7334    1219    2494    85.75   74.62   79.80  Bu noktada spanleri duzelttim