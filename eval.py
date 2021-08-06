from pathlib import Path

from semantictagger import conllu
from semantictagger.conllu import CoNLL_U
from semantictagger.dataset import Dataset
from semantictagger.paradigms import Encoder

from typing import Dict, Generator, Iterator , Tuple , Union
import os 
from pandas import DataFrame


class EvaluationModule():

    def __init__(self, 
        paradigm : Encoder , 
        dataset : Dataset , 
        pathroles : Union[Path,str] , 
        pathpredicates : Union[Path,str] , 
        span_based = True,
        mockevaluation : bool = False , 
        ):

        self.paradigm : Encoder = paradigm
        self.dataset : Dataset = dataset
        self.span_based : bool = True


        self.mockevaluation = mockevaluation

        if not self.mockevaluation :
            self.rolesgen : Iterator = iter(self.__readresults__(pathroles))
            self.predgen : Iterator = iter(self.__readresults__(pathpredicates))
       
        self.entryiter : Iterator = iter(self.dataset)
    
    def __readresults__(self , path):
        """
            Flair outputs two files , called dev.tsv and test.tsv respectively.
            These files can be read with this function and each time this function 
            will yield one entry, which in theory should align with the corresponding dataset.
        """
        entryid = 0
        entry = ["" for d in range(100)]
        counter = 0 

        with path.open() as f:
            while True:
                line = f.readline().replace("\n" , "")

                if line is None:
                    break

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

        return None

    
    def single(self , verbose = False):
        
        try:
            target = next(self.entryiter)
        except StopIteration:
            return None
        
        
        words = target.get_words()


        if not self.mockevaluation:
            preds = next(self.predgen)
            if preds is None:
                return None

            roles = next(self.rolesgen)        
            predicted = self.paradigm.to_conllu(words , preds , roles)

        else :
            roles = self.paradigm.encode(target)
            preds = ["V" if x != "" and x!="_" else "_" for x in target.get_vsa()]
            predicted = self.paradigm.to_conllu(words , preds , roles)

        
        if verbose:
            
            a = {"words":words , "predicates" : preds , "roles" : roles}
            a.update({f"TARGET {i}" : v for i  , v in enumerate(target.get_span())})
            a.update({f"PRED {i}" : v for i  , v in enumerate(predicted.get_span())})

            return DataFrame(a)
        

        return target , predicted


    def createpropsfiles(self, saveloc = "./evaluation/conll05"):

        for i in ["pred.props" , "target.props"]:
            if os.path.isfile(os.path.join(saveloc , i)):
                print("CreatePropsFiles : Removing old .props files...")
                os.remove(os.path.join(saveloc , i))


        with open(os.path.join(saveloc , "pred.props" ) , "x") as fp:
            with open(os.path.join(saveloc , "target.props") , "x") as ft:
                while True:
                    
                    s = self.single()
                    
                    if s is None :
                        return 
                    
                    target , predicted = s[0] , s[1]
                    
                    vsa = predicted.get_vsa()
                    vsa = ["V" if a != "_" and a != "" else "-" for a in vsa]
                    spans = predicted.get_span()

                    for i in range(len(vsa)):
                        fp.write(f"{vsa[i]}\t")
                        for j in range(len(spans)):
                            fp.write(f"{spans[j][i]}\t")
                        fp.write("\n")
                    fp.write("\n")

                    vsa = target.get_vsa()
                    vsa = ["V" if a != "_" else "-" for a in vsa]
                    spans = target.get_span()

                    for i in range(len(vsa)):
                        ft.write(f"{vsa[i]}\t")
                        for j in range(len(spans)):
                            ft.write(f"{spans[j][i]}\t")
                        ft.write("\n")
                    ft.write("\n")