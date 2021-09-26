from pathlib import Path, PureWindowsPath

from semantictagger import conllu
from semantictagger import paradigms
from semantictagger.conllu import CoNLL_U
from semantictagger.dataset import Dataset
from semantictagger.paradigms import Encoder

from typing import Dict, Generator, Iterator , Tuple , Union
import os 
from pandas import DataFrame
from tqdm import tqdm
import pdb 
class EvaluationModule():

    def __init__(self, 
        paradigm : Encoder , 
        dataset : Dataset , 
        early_stopping : int,
        pathroles : Union[Path,str] , 
        path_frame_file : Union[Path,str] ,
        path_pos_file : Union[Path,str],
        goldframes : bool = False,
        goldpos : bool = False,
        span_based = True,
        mockevaluation : bool = False , 
        ):
        
        self.postype : paradigms.POSTYPE = paradigm.postype 
        self.early_stopping = early_stopping
        self.paradigm : Encoder = paradigm
        self.dataset : Dataset = dataset
        self.span_based : bool = span_based
        self.mockevaluation = mockevaluation
        self.goldpos = goldpos
        self.goldframes = goldframes

        if not self.mockevaluation :
            self.rolesgen : Iterator = iter(self.__readresults__(pathroles , gold = False))
            self.predgen : Iterator = iter(self.__readresults__(path_frame_file, gold = goldframes))
            self.posgen : Iterator = iter(self.__readresults__(path_pos_file , gold = goldpos))

       
        self.entryiter : Iterator = iter(self.dataset)
    
    def __readresults__(self , path , gold):
        """
            Flair outputs two files , called dev.tsv and test.tsv respectively.
            These files can be read with this function and each time this function 
            will yield one entry, which in theory should align with the corresponding dataset.
        """
        entryid = 0
        entry = ["" for d in range(100)]
        counter = 0 
        earlystoppingcounter = 0

        
        if type(path) == str:
            path = Path(path)

        with path.open() as f:
            while True:
                line = f.readline().replace("\n" , "").replace("\t" , " ")

                if line is None:
                    break

                if line == "" : 
                    entryid += 1
                    yield entry[:counter]
                    entry = ["" for d in range(100)] 
                    counter = 0
                    earlystoppingcounter +=1
                    if self.early_stopping != False:
                        if self.early_stopping == earlystoppingcounter :
                            return None
                else : 
                    
                    elems = line.split(" ")
                    if len(elems) == 1: 
                        entry[counter] = ""
                    elif len(elems) == 2:
                        if gold :
                            entry[counter] = elems[1]
                        else :
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
            pos = next(self.posgen)
            if preds is None:
                return None
            preds = ["V" if x != "" and x!= "_" else "_" for x in preds]
            roles = next(self.rolesgen)
            if roles is None:
                return None     
            predicted = self.paradigm.spanize(words , preds , roles , pos)

        else :
            roles = self.paradigm.encode(target)
            if self.postype == paradigms.POSTYPE.UPOS:
                pos = target.get_by_tag("upos")
            else :
                pos = target.get_by_tag("xpos")

            preds = ["V" if x != "" and x!= "_" else "_" for x in target.get_vsa()]
            predicted = self.paradigm.spanize(words , preds , roles , pos)

        
        if verbose:
            
            a = {"words":words , "predicates" : preds , "roles" : roles}
            a.update({f"TARGET {i}" : v for i  , v in enumerate(target.get_span())})
            a.update({f"PRED {i}" : v for i  , v in enumerate(predicted.get_span())})

            return DataFrame(a)
    

        return target , predicted , roles

    def createpropsfiles(self, saveloc , debug = False):

        for i in ["predicted-props.tsv" , "target-props.tsv"]:
            if os.path.isfile(os.path.join(saveloc , i)):
                print("CreatePropsFiles : Removing old .props files...")
                os.remove(os.path.join(saveloc , i))

        counter = -1

        with open(os.path.join(saveloc , "predicted-props.tsv" ) , "x") as fp:
            with open(os.path.join(saveloc , "target-props.tsv") , "x") as ft:
                total = len(self.dataset)
                if self.early_stopping != False:
                    total = min(len(self.dataset),self.early_stopping)
                for i in tqdm(range(total)):
                    s = self.single()
                    
                    if s is None :
                        return 
                    
                    target , predicted , roles = s[0] , s[1] , s[2]
                    
                    files = (fp , ft)
                    entries = (predicted , target)
                    counter += 1 

                    for k in range(2):                     
                        if self.span_based:
                            if k == 0 :
                                spans = predicted
                            else:
                                spans = entries[k].get_span()
                        else :
                            spans = entries[k].get_depbased()
                        
                        vsa = entries[1].get_vsa()
                        vsa = ["V" if a != "_" and a != "" else "-" for a in vsa]
                        words = entries[1].get_words()
                        
                        if debug:
                            files[k].write(f"{counter}\n")

                        for i in range(len(vsa)):
                            
                            if debug:
                                files[k].write(f"{words[i]}\t")
                                files[k].write(f"{roles[i]}\t")

                            files[k].write(f"{vsa[i]}\t")
                        


                            for j in range(len(spans)):
                                files[k].write(f"{spans[j][i]}\t")
                            files[k].write("\n")
                        files[k].write("\n")

