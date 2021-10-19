
from pathlib import Path, PureWindowsPath
from numpy.lib.function_base import select

from semantictagger import conllu
from semantictagger import paradigms
from semantictagger.conllu import CoNLL_U
from semantictagger.dataset import Dataset
from semantictagger.paradigms import Encoder, ParameterError , RELPOSVERSIONS
from semantictagger.datatypes import Outformat

from typing import Dict, Generator, Iterator , Tuple , Union
import os 
from pandas import DataFrame
from tqdm import tqdm
import pdb 
import enum
import numpy as np
import re

import pdb
import uuid
import matplotlib.pyplot as plt

class RoleTypes(enum.Enum):
    MISSED= 0 
    CORRECT = 1
    FALSE = 2


class ResultSet():
    def __init__(self):
        self.distance_dict = {}
        for i in range(15):
            self.distance_dict[i]= {
                "Correct":0 , 
                "Missed Role":0,
                "False Role":0,
            }

        self.syntax_semantics_dict ={
            "Correct" :0 ,
            "False" : 0
        }

        self.correct_index = {}
        self.missed_role_index ={}
        self.false_role_index ={}
        self.target_tags_count = {}
        self.target_tags_counter = 0 

        self.search_for_confusions = ["-1,ARG2|-1,ARG1" ,"-1,ARG1|-1,ARG2","1,ARG1|1,ARG0","-1,ARG3|-1,ARG2","1,ARGM-MNR|1,ARGM-ADJ",""]


    def add_distance(self,
        dist,
        roletype:RoleTypes,
        ):   

        if roletype ==  RoleTypes.CORRECT:
            key = "Correct" 
        elif roletype == RoleTypes.FALSE:
            key = "False Role"
        elif roletype == RoleTypes.MISSED:
            key = "Missed Role"
        else :
            Exception()

        self.distance_dict[abs(dist)][key] += 1

    def add_confusion(self,pdist,prole,tdist,trole):    
        self.add_distance(tdist,roletype=RoleTypes.FALSE)

        if f"{pdist},{prole}|{tdist},{trole}" in self.false_role_index:
            self.false_role_index[f"{pdist},{prole}|{tdist},{trole}"] += 1
        else :
            self.false_role_index[f"{pdist},{prole}|{tdist},{trole}"] = 1
    
    def add_correct_index(self,pdist,prole):
        self.add_distance(pdist,roletype=RoleTypes.CORRECT)

        if f"{pdist},{prole}" in self.correct_index:
            self.correct_index[f"{pdist},{prole}"] += 1
        else :
            self.correct_index[f"{pdist},{prole}"] = 1

    def add_missed_index(self,tdist,trole):
        self.add_distance(tdist,roletype=RoleTypes.MISSED)
        if f"{tdist},{trole}" in self.missed_role_index:
            self.missed_role_index[f"{tdist},{trole}"] += 1
        else :
            self.missed_role_index[f"{tdist},{trole}"] = 1

    def add_target_tag(self,tdist,trole):
        if f"{tdist},{trole}" in self.target_tags_count:
            self.target_tags_count[f"{tdist},{trole}"] += 1
        else :
            self.target_tags_count[f"{tdist},{trole}"] = 1
        
        self.target_tags_counter += 1

    def add_syntactic(self, distance , Tdistance , postag , Tpostag , deprel , Tdeprel):
        if distance == Tdistance and postag == Tpostag and deprel == Tdeprel:
            self.syntax_semantics_dict["Correct"] +=1
        else :
            self.syntax_semantics_dict["False"] +=1
            
    def get_distance_dict(self):
        self.distance_dict.pop(0)
        self.distance_dict.pop(12)
        self.distance_dict.pop(13)
        self.distance_dict.pop(14)
        return self.distance_dict
            
    def get_other_dicts(self,get_first=20):
        self.correct_index = sorted([(x[0],x[1],"{:0.2f}".format(x[1]/self.target_tags_count[x[0]])) for x in self.correct_index.items()],key = lambda x : x[2],reverse=True)[:get_first]
        self.false_role_index = sorted([(x[0].split("|")[0],x[0].split("|")[1],x[1]) for x in self.false_role_index.items()], key=lambda z:z[2], reverse=True)[:get_first]
        self.missed_role_index = sorted([(x[0],x[1],"{:0.2f}".format(x[1]/self.target_tags_count[x[0]])) for x in self.missed_role_index.items()], key= lambda x:x[1],reverse=True)[:get_first]
        
        return self.correct_index,self.false_role_index,self.missed_role_index




class EvaluationModule():

    def __init__(self, 
        paradigm : Encoder , 
        dataset : Dataset , 
        pathroles : Union[Path,str]=None, 
        path_frame_file : Union[Path,str]=None ,
        path_pos_file : Union[Path,str] =None,
        goldframes : bool = False,
        goldpos : bool = False,
        mockevaluation : bool = False , 
        early_stopping : bool = False
        ):
        
        self.early_stopping = early_stopping
        self.postype : paradigms.POSTYPE = paradigm.postype 
        self.paradigm : Encoder = paradigm
        self.dataset : Dataset = dataset
        self.mockevaluation = mockevaluation
        self.goldpos = goldpos
        self.goldframes = goldframes

        self.pathroles = pathroles
        self.path_frame_file = path_frame_file
        self.path_pos_file = path_pos_file
        self.goldframes = goldframes
        self.goldpos = goldpos

        if not self.mockevaluation :
            if pathroles is None or path_frame_file is None or path_pos_file is None:
                ParameterError("Initializing EvaluationModule without mockevaluation feature, requires path/to/rolefile path/to/frame/file and path/to/pos/file.")

            self.rolesgen : Iterator = iter(self.__readresults__(pathroles , gold = False))
            self.predgen : Iterator = iter(self.__readresults__(path_frame_file, gold = goldframes))
            self.posgen : Iterator = iter(self.__readresults__(path_pos_file , gold = goldpos))

       
        self.entryiter : Iterator = iter(self.dataset)
    

    def reset_buffer(self):
        self.rolesgen : Iterator = iter(self.__readresults__(self.pathroles , gold = False))
        self.predgen : Iterator = iter(self.__readresults__(self.path_frame_file, gold = self.goldframes))
        self.posgen : Iterator = iter(self.__readresults__(self.path_pos_file , gold = self.goldpos))
        self.entryiter : Iterator = iter(self.dataset)


    def __readresults__(self , path , gold , getpair=False):
        """
            Flair outputs two files , called dev.tsv and test.tsv respectively.
            These files can be read with this function and each time this function 
            will yield one entry, which in theory should align with the corresponding dataset.
        """
        entryid = 0
        entry = ["" for d in range(100)]
        pairentry = ["" for d in range(100)]
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
                    if getpair:
                        yield (entry[:counter] , pairentry[:counter])
                    else :
                        yield entry[:counter]
                    entry = ["" for d in range(100)] 
                    pairentry = ["" for d in range(100)]

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
                        if getpair:
                            pairentry[counter] = elems[1]
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
            # preds = ["V" if x != "" and x!= "_" else "_" for x in preds]
            roles = next(self.rolesgen)
            if roles is None:
                return None     
            predicted = self.paradigm.to_conllu(words , preds , roles , pos)

        else :
            roles = self.paradigm.encode(target)
            if self.postype == paradigms.POSTYPE.UPOS:
                pos = target.get_by_tag("upos")
            else :
                pos = target.get_by_tag("xpos")
            
            preds = target.get_vsa()
            # preds = ["V" if x != "" and x!= "_" else "_" for x in target.get_vsa()]
            predicted = self.paradigm.to_conllu(words , preds , roles , pos)

        
        if verbose:
            
            a = {"words":words , "predicates" : preds , "roles" : roles}
            a.update({f"TARGET {i}" : v for i  , v in enumerate(target.get_span())})
            a.update({f"PRED {i}" : v for i  , v in enumerate(predicted.reconstruct())})

            return DataFrame(a)
    

        return target , predicted , roles

    def createpropsfiles(self, saveloc , depth_constraint = False, debug = False):

        counter = -1

        outformats = [
            Outformat.CONLL05,
            Outformat.CONLL09
        ]
        
        with open(os.path.join(saveloc , "predicted-props.tsv") , "x") as fp:
            with open(os.path.join(saveloc ,"target-props.tsv") , "x") as ft:
                with open(os.path.join(saveloc , "predicted-props-conll09.tsv") , "x") as fp1:
                    with open(os.path.join(saveloc ,"target-props-conll09.tsv") , "x") as ft1:
                        total = len(self.dataset)
                        if self.early_stopping != False:
                            total = min(len(self.dataset),self.early_stopping)
                        for i in tqdm(range(total)):
                            s = self.single()
                            
                            if s is None :
                                return 
                            
                            target , predicted , roles = s[0] , s[1] , s[2]
                            
                            files = [(fp , ft),(fp1,ft1)]
                            entries : Tuple[CoNLL_U]= (predicted , target)
                            if depth_constraint:
                                if target.depth != depth_constraint:
                                    continue

                            counter += 1 
                            for fpindex , fpval in enumerate(outformats):
                                for k in range(2):                     
                                    if fpval == Outformat.CONLL05:
                                        if k == 0 :
                                            spans = self.paradigm.reconstruct(entries[k])
                                        else:
                                            spans = entries[k].get_span()
                                    # elif of[0] == Outformat.CoNLL09:
                                    else:
                                        if k == 0:
                                            files[fpindex][k].write(entries[k].to_conll_text(self.paradigm.frametype, entries[1]))
                                        else :
                                            files[fpindex][k].write(entries[k].to_conll_text(self.paradigm.frametype))
                                        continue

                                    vsa = entries[1].get_vsa()
                                    if fpval == Outformat.CONLL05:
                                        vsa = ["V" if a != "_" and a != "" else "-" for a in vsa]
                                    words = entries[1].get_words()
                                    
                                    if debug:
                                        files[fpindex][k].write(f"{counter}\n")

                                    for i in range(len(vsa)):
                                        
                                        if debug:
                                            files[fpindex][k].write(f"{words[i]}\t")
                                            files[fpindex][k].write(f"{roles[i]}\t")

                                        files[fpindex][k].write(f"{vsa[i]}\t")

                                        for j in range(len(spans)):
                                            files[fpindex][k].write(f"{spans[j][i]}\t")
                                        files[fpindex][k].write("\n")
                                    files[fpindex][k].write("\n")

    def evaluate(self , path , depth_constraint=False , automated_testing=False):
        self.reset_buffer()
        self.createpropsfiles(path, depth_constraint=depth_constraint)
        conll09 = self.__evaluate_conll09(path,automated_testing)
        conll05 = self.__evaluate_conll05(path,automated_testing)
        

        return { 
            "CoNLL05" : conll05
        }

    def create_conllu_files(self,path):
         with open(os.path.join(path , "predicted.conllu") , "x") as fp:
            with open(os.path.join(path ,"target.conllu") , "x") as ft:
                for entry in self.dataset:
                    ft.write(str(entry))
                    predicted = self.paradigm.to_conllu(entry.get_words(),entry.get_vsa(),self.paradigm.encode(entry),entry.get_pos(self.paradigm.postype))
                    fp.write(predicted.__str__(entry))

    def __evaluate_conll09(self,path, automated_testing=False):
        if automated_testing:return
        # TODO add quiet -q.
        os.popen(f'perl ./evaluation/conll09/eval09.pl -g {path}/target-props-conll09.tsv -s {path}/predicted-props-conll09.tsv > {path}/conll09-results.txt')


    def __evaluate_conll05(self,path,automated_testing = False):
        
        sh_command = f'perl ./evaluation/conll05/srl-eval.pl  ./{path}/target-props.tsv ./{path}/predicted-props.tsv'
        if not automated_testing:
            sh_command = f'perl ./evaluation/conll05/srl-eval.pl -latex -C ./{path}/target-props.tsv ./{path}/predicted-props.tsv  > ./{path}/conll05-results.txt'

        with os.popen(sh_command) as output:
            while True:
                line = output.readline()
                print(line)
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
                    return results
        
        return None

    def mockevaluate(self):
        path = f"mockevals/{str(uuid.uuid1())[:5]}"
        print(f"Mock evaluation path : {path}")
        os.makedirs(f"{path}")
        self.evaluate(path)
        self.create_conllu_files(path)

    def reevaluate(self,path , depth_constraint = False , graph_for_depth=False):
        if graph_for_depth:
            precisions = []
            recalls = []
            for j in range(1,5):
                for i in ["target-props-conll09.tsv","predicted-props-conll09.tsv","target-props.tsv","predicted-props.tsv","conll09-results.txt","conll05-results.txt"]:
                    if os.path.isfile(os.path.join(path,i)):
                        os.remove(os.path.join(path,i))
                result_dict = self.evaluate(path , depth_constraint=j,automated_testing=True)
                precisions.append(result_dict["CoNLL05"]["precision"])
                recalls.append(result_dict["CoNLL05"]["recall"])

            # print(results)
            plt.plot(['1','2','3','4'],precisions,recalls)
            plt.legend(["Recall","Precision"])
            plt.title("P/R by depth")
            plt.ylabel('Percentage')
            plt.xlabel('Depth')
            plt.savefig(os.path.join(path,"by_depth.png"))
            return
        
        
        for i in ["target-props-conll09.tsv","predicted-props-conll09.tsv","target-props.tsv","predicted-props.tsv","conll09-results.txt","conll05-results.txt"]:
            if os.path.isfile(os.path.join(path,i)):
                os.remove(os.path.join(path,i))
        self.evaluate(path , depth_constraint)

        


    def role_prediction_by_distance(self , depth_constraint = False , latex = True):
        self.rolesgen = iter(self.__readresults__(self.pathroles , gold = False,getpair=True))
        self.entryiter : Iterator = iter(self.dataset)
        tagger_version = self.paradigm.version
        
        rset = ResultSet()


        


        while True:
            a , b  = next(self.rolesgen)
            entry  = next(self.entryiter)
            
            if entry is None:
                break
            
            # if entry.get_sentence().startswith("Let me join the choru") or "I 'm not fond of the Google - hates - privacy argument" in entry.get_sentence(): 
            #     print(entry.get_sentence())
            # else :
            #     continue

            if depth_constraint != False:
                if len(entry.get_srl_annotation()) != depth_constraint:
                    continue


            srl = [*zip(*entry.get_srl_annotation())]
            vlocs = entry.get_verb_indices()
            postags = entry.get_pos(self.postype)
            
            if a is None:
                break
           
            for i in range(len(a)):
                t =  a[i].split(",")
                s =  b[i].split(",")
                if tagger_version == RELPOSVERSIONS.SRLEXTENDED:
                    if len(s) == 5:
                        distance , postag , deprel , preddistance , predrole = int(s[0]),s[1],s[2],int(s[3]),s[4]
                        rset.add_target_tag(s[3],s[4])
                        
                        if len(t) != 5:
                            rset.add_missed_index(preddistance,predrole)
                        else :
                            Tdistance , Tpostag , Tdeprel , Tpreddistance , Tpredrole = int(t[0]),t[1],t[2],int(t[3]),t[4]
                            if preddistance == Tpreddistance and predrole == Tpredrole:
                                if depth_constraint: assert(preddistance == depth_constraint)
                                rset.add_correct_index(preddistance,predrole)
                            elif preddistance == Tpreddistance and predrole != Tpredrole:
                                rset.add_syntactic(preddistance,Tpreddistance,postag,Tpostag,deprel,Tdeprel)
                                rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                            else :
                                layer = srl[i]
                                if len([1 for k in layer if k != "_" and k !="V"]) < 2 : 
                                    rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                                    continue
                                if Tpredrole in layer:
                                    pointeddepth = 0 
                                    if i < vlocs[0] : pointeddepth = Tpreddistance-1 
                                    elif i == vlocs[0] : pointeddepth = Tpreddistance
                                    elif  i > vlocs[-1] : pointeddepth = len(vlocs) - Tpreddistance
                                    elif i == vlocs[-1] : pointeddepth = len(vlocs) - Tpreddistance - 1
                                    else : 
                                        tempind = 0 
                                        for verbindex in vlocs:
                                            if verbindex < i:
                                                tempind += 1
                                            elif verbindex == i:
                                                pointeddepth = tempind + (Tpreddistance  if Tpreddistance > 0 else -Tpreddistance)
                                                break
                                            else:
                                                pointeddepth = tempind + (Tpreddistance - 1 if Tpreddistance > 0  else -Tpreddistance)
                                                break
                                    
                                    if pointeddepth < 0 or pointeddepth >= len(layer):
                                        rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                                        continue
                                    if layer[pointeddepth] == Tpredrole:
                                        if depth_constraint: assert(preddistance == depth_constraint)
                                        rset.add_correct_index(preddistance,predrole)
                                    else :
                                        rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                                else :
                                    rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                                    continue

                elif tagger_version == RELPOSVERSIONS.SRLREPLACED:
                    distance , postag , deprel = int(s[0]),s[1],s[2]
                    if postag != "FRAME": continue
                    if len(t) < 2:
                        if postag == "FRAME":
                            rset.add_target_tag(distance,deprel)
                            rset.add_missed_index(distance,deprel)
                        continue

                    Tdistance , Tpostag , Tdeprel = int(t[0]),t[1],t[2]
                    rset.add_target_tag(distance,deprel)
                    
                    if Tpostag != "FRAME": 
                        rset.add_missed_index(distance,deprel)
                    else :
                        preddistance = distance
                        Tpreddistance = Tdistance
                        predrole = deprel
                        Tpredrole = Tdeprel
                        
                        if preddistance == Tpreddistance and predrole == Tpredrole:
                            rset.add_correct_index(preddistance,predrole)
                        elif preddistance == Tpreddistance and predrole != Tpredrole:
                            rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                        else :
                            layer = srl[i]
                            if len([1 for k in layer if k != "_" and k !="V"]) < 2 : 
                                rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                                continue
                            if Tpredrole in layer:
                                pointeddepth = 0 
                                if i < vlocs[0] : pointeddepth = Tpreddistance-1 
                                elif i == vlocs[0] : pointeddepth = Tpreddistance
                                elif  i > vlocs[-1] : pointeddepth = len(vlocs) - Tpreddistance
                                elif i == vlocs[-1] : pointeddepth = len(vlocs) - Tpreddistance - 1
                                else : 
                                    tempind = 0 
                                    for verbindex in vlocs:
                                        if verbindex < i:
                                            tempind += 1
                                        elif verbindex == i:
                                            pointeddepth = tempind + (Tpreddistance  if Tpreddistance > 0 else -Tpreddistance)
                                            break
                                        else:
                                            pointeddepth = tempind + (Tpreddistance - 1 if Tpreddistance > 0  else -Tpreddistance)
                                            break
                                
                                if pointeddepth < 0 or pointeddepth >= len(layer):
                                    rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                                    continue
                                if layer[pointeddepth] == Tpredrole:
                                    rset.add_correct_index(preddistance,predrole)
                                else :
                                    rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                            else :
                                rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)

                elif tagger_version == RELPOSVERSIONS.DEPLESS or tagger_version == RELPOSVERSIONS.FLATTENED:
                    if len(s) != 3 : continue
                    preddistance , postag , predrole = int(s[0]),s[1],s[2]
                    
                    temparray= [preddistance]
                    temparray2 = [postag]
                    pointedverblocs = [0]
                    for c in range(1):
                        start = i-1 if temparray[c] < 0 else i+1
                        end = -1 if temparray[c] < 0 else len(postags)
                        seen = 0 
                        vdistance = 0 
                        for j in range(start , end , np.sign(temparray[c])):
                            if j in vlocs:
                                vdistance += 1
                            if postags[j] == temparray2[c]:
                                seen+=1
                                if seen == np.abs(temparray[c]):
                                    pointedverblocs[c] = vdistance
                                    break
                    preddistance = pointedverblocs[0] * np.sign(temparray[c])
                    # yield preddistance , predrole , entry.get_sentence() , i 
                    # if preddistance > 1 :
                    #     print(DataFrame({"Words":entry.get_words() , "Encoded" :b ,"POS":postags , "Verbs":["V" if x in vlocs else "_" for x in range(len(entry))], "Ilgili":["" if x != i else f"{preddistance}" for x in range(len(entry))]}))
                    #     print("\n\n")


                    assert(preddistance != 0 )
                    if depth_constraint :
                        assert(preddistance <= depth_constraint )

                    rset.add_target_tag(preddistance,predrole)

                    if len(t) != 3 : 
                        rset.add_missed_index(preddistance,predrole)
                    else :

                        Tpreddistance , Tpostag , Tpredrole = int(t[0]),t[1],t[2]
                        temparray= [Tpreddistance]
                        temparray2 = [Tpostag]
                        pointedverblocs = [0]
                        for c in range(1):
                            start = i-1 if temparray[c] < 0 else i+1
                            end = -1 if temparray[c] < 0 else len(postags)
                            seen = 0 
                            vdistance = 0 
                            for j in range(start , end , np.sign(temparray[c])):
                                if j in vlocs:    
                                    vdistance += 1
                                if postags[j] == temparray2[c]:
                                    seen+=1   
                                    if seen == np.abs(temparray[c]):
                                        pointedverblocs[c] = vdistance
                                        break
                            
                        Tpreddistance = pointedverblocs[0] * np.sign(temparray[c])
                        

                        if preddistance == Tpreddistance and predrole == Tpredrole:
                            rset.add_correct_index(preddistance,predrole)
                        elif preddistance == Tpreddistance and predrole != Tpredrole:
                            rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)

                        else :
                            layer = srl[i]
                            try :
                                if layer[Tpreddistance] == Tpredrole:
                                    if layer[pointeddepth] == Tpredrole:
                                        rset.add_correct_index(preddistance,predrole)
                                    else :
                                        rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)
                            except:
                                rset.add_confusion(preddistance,predrole,Tpreddistance,Tpredrole)    
                else:
                    continue


            
           
            

        print(rset.target_tags_counter , " viele Role Woerter insgesamt.")
    

        print(DataFrame(rset.get_distance_dict()).to_latex())
        temparray = rset.get_other_dicts()
        print(DataFrame(temparray[0]).to_latex(caption="Correct roles."))
        print(DataFrame(temparray[1]).to_latex(caption="Missed roles."))
        print(DataFrame(temparray[2]).to_latex(caption="Confusion of roles."))

        # print(DataFrame(false_role_list).to_latex(caption="False labels."))
        print(DataFrame(sorted(rset.target_tags_count.items(),key=lambda x : x[1],reverse = True)[:20]).to_latex(caption="Distribution of semantic labels."))
        print(rset.syntax_semantics_dict)



 

    def inspect_learning_behavior(self,path,num_of_epoch :int , plot = True):
        
        syntax_correct_semantic_false = []
        syntax_correct_semantic_correct = []
        syntax_false_semantic_correct = []
        syntax_false_semantic_false = []

        for i in range(1,num_of_epoch+1):
            with open(f"{path}/dev{i}.tsv") as fp:
                typ1 = 0
                typ2 = 0 
                typ3 = 0
                typ4 = 0 

                while True:
                    line = fp.readline()
                    if len(line) == 0:
                        syntax_correct_semantic_false.append(typ1)
                        syntax_correct_semantic_correct.append(typ2)
                        syntax_false_semantic_correct.append(typ3)
                        syntax_false_semantic_false.append(typ4)
                        break
                    if line == "\n":
                        continue
                    
                    line = line.replace("\n","")
                    array = line.split(" ")
                    word , pred , target = array[0] , array[1] , array[2]
                    if self.paradigm.version == RELPOSVERSIONS.FLATTENED:
                        t1 = array[1].split(",")
                        t2 = array[2].split(",")
                        

                        if t1[0] != t2[0] or  t1[1] != t2[1]:
                            if len(t1) == 3 and len(t2) == 3:
                                if t1[2] == t2[2]:
                                    # print(t1 , t2 , "syntax_false_semantic_correct")
                                    typ3 += 1
                                else :
                                    # print(t1 , t2 , "syntax_false_semantic_false")
                                    typ4 += 1
                            elif len(t1) == 2 and len(t2) == 3 or len(t1) == 3 and len(t2) == 2:
                                # print(t1 , t2 , "syntax_false_semantic_false")
                                typ4 += 1
                            else:
                                # print(t1 , t2 , "syntax_false_semantic_correct")
                                typ3 += 1
                        else :
                            if len(t1) == 3 and len(t2) == 3:
                                if t1[2] == t2[2]:
                                    # print(t1 , t2 , "syntax_correct_semantic_correct")
                                    typ2 += 1
                                else :
                                    # print(t1 , t2 , "syntax_correct_semantic_false")
                                    typ1 += 1
                            elif len(t1) == 2 and len(t2) == 3 or len(t1) == 3 and len(t2) == 2:
                                # print(t1 , t2 , "syntax_correct_semantic_false")
                                typ1 += 1
                            else:
                                # print(t1 , t2 , "syntax_correct_semantic_correct")
                                typ2 += 1





                    # elif self.paradigm.version == RELPOSVERSIONS.SRLEXTENDED:
                    # else:
        if plot:
            import matplotlib.pyplot as plt
    
            plt.subplot(121)
            plt.title('Semantic Error')
            plt.plot(syntax_correct_semantic_false)
            plt.ylabel("Occurence")
            plt.xlabel("Epoch")
            plt.subplot(122)
            plt.title('Syntactic Error')
            plt.plot(syntax_false_semantic_correct)
            plt.ylabel("Occurence")
            plt.xlabel("Epoch")
            # plt.subplot(223)
            # plt.title('Both Correct')
            # plt.plot(syntax_correct_semantic_correct)
            # plt.ylabel("Occurence")
            # plt.xlabel("Epoch")
            # plt.subplot(224)
            # plt.title('Both False')
            # plt.plot(syntax_false_semantic_false)
            # plt.ylabel("Occurence")
            # plt.xlabel("Epoch")
            plt.show()
        return syntax_correct_semantic_false , syntax_correct_semantic_correct , syntax_false_semantic_correct , syntax_false_semantic_false