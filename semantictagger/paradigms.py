import abc
import re

from . import conllu
from . import dataset 
from typing import Dict, Mapping , Union , List , Tuple 
import pdb 


EMPTY_LABEL = "_"

class MapProtocol():
    def __init__(self ,lowkey : int = -1 , unkwn : 'str' = "<UNKNW>"):
        self.lowkey = lowkey
        self.unkwn = unkwn

class Mapper():
    def __init__(self ,datasets : Union[dataset.Dataset, List[dataset.Dataset]] , 
        encoder , 
        protocol : MapProtocol,
        percentage : int =  5 , 
        lowerbound : int = -1):
        
        if type(datasets) == dataset.Dataset:
            datasets = [datasets]
        
        self.datasets = datasets
        self.encoder = encoder
        self.protocol = protocol
        self.percentage = percentage
        self.lowerbound = lowerbound
        self.usebound = True if self.lowerbound != -1 else False
        self.mapping = {}
        self.initmaptag()

    def initmaptag(self): 

        for dataset in self.datasets:
            for sentence in dataset:
                encoded = self.encoder.encode(sentence)
                for i in encoded : 
                    if not i in self.mapping :
                        self.mapping[i] = 1 
                    else :
                        self.mapping[i] += 1

        if not self.usebound:
            medval =  sum(self.mapping.values()) // len(self.mapping)
            minval = min(self.mapping.values())

            fiftyper = medval - minval
            self.lowerbound = fiftyper // (50 // self.percentage)
        
        print(f"Discarding all elements which have less than {self.lowerbound} occurences.")

        for i in self.mapping.keys():
            if self.mapping[i] <= self.lowerbound :
                self.mapping[i] = self.protocol.lowkey
            else : 
                continue
    
    def maptag(self,tag : str):
        if self.mapping[tag] != self.protocol.lowkey:
            return tag

        return tag.split("|")[2]


class ParameterError(Exception):
    def __init__(self):
        super().__init__("False Parameter Error.")


class PairMapper:
    """
        A simple class to keep things ordered.
        Some roles occur at the same time quite often.
        To see the exact numbers please refer to 
        dataset.Dataset.findcommonroles 

    """
    def __init__(self , roles : List[Tuple[str,str,str]]):
        self.usedcount = 0 
        self.roles = roles

    def map(self , ann : List[str]) -> Union[str,bool]:
        for i in range(len(ann)):
            for j in range(i+1,len(ann)):
                for role in self.roles:
                    if role[0] == ann[i] and role[1]==ann[j]:
                        self.usedcount += 1
                        return role[2]
        
        return False

    def __str__(self) -> str:

        _str = "PAIRMAP \n"
        for i in self.roles:
            _str += f"\t {i[0]} , {i[1]} -> {i[2]} \n"
        _str +=  f"PairMapper paired {self.usedcount} times."

        return _str



class Encoder(abc.ABC):
    
    @abc.abstractclassmethod
    def encode(self,entry : conllu.CoNLL_U ) -> List[str]:
        pass

    @abc.abstractclassmethod
    def decode(self,encoded : List[str]) -> List[List[str]]:
        pass

    # TODO 
    # @abc.abstractclassmethod
    # def spanize(self , List[str], ) -> List[List[str]]:
    #     encoded = self.encode(entry)

    def test(self, entry : Union[conllu.CoNLL_U  , Tuple[conllu.CoNLL_U , List[str],List[str]]]) -> Tuple[int,int]:
        """
            Checks if an entry is correctly tagged by looking at
            whether it could be successfully converted back into 
            CoNLL_U.get_srl_annotation(self , depth) format.
        """
        if type(entry) == conllu.CoNLL_U:
            vlocs = ["V" if x != "_" and x != "" else "" for x in entry.get_vsa()]
            encoded = self.encode(entry)
            cmp = [entry.get_srl_annotation(d) for d in range(entry.depth)]

        else :
            vlocs = entry[1]
            encoded = entry[2]
            cmp = [entry[0].get_srl_annotation(d) for d in range(entry[0].depth)]

        
        decoded = self.decode(encoded , vlocs)
    
        
        correct = 0
        false = 0 

        try:
            for i in range(len(cmp)):
                for j in range(len(cmp[i])):
                    if cmp[i][j] == decoded[i][j]:
                        correct += 1
                    else :
                        false += 1
        except IndexError:
            return correct , false + (len(cmp) - len(decoded))*(len(cmp[0]))
        
       
        return correct , false

class DIRECTTAG(Encoder):
    """
        Directional Encoder. Every word is responsible for holding its 
        own tag. To show that a non-predicate is connected to a predicate,  
        '>' or '<' , depending on predicates relative location , are used.
        '>' simply means next predicate to the right.
        
        Multiple uses are allowed  : e.g '>>ARG0'
        This multiplicity can be limited by mult arg of this class. 
    """

    def __init__(self , 
            mult, 
            rolehandler : str = 'complete',
            verbshandler : str = 'complete',
            verbsonly = False,
            deprel = False,
            depreldepth = 1,
            pairmap : PairMapper = None
            ):
        """
        :param mult : Designates the maximum amount of concatenation of either '>' or '<' to a role word.
        :param verbshandled : How verbs are treated by the encoder. 
            Default: 'complete' is set then a verb notation such as love.02 is used as is.
            if 'omitlemma' is set then a verb notation such as love.02 is reduced to V02
            elif 'omitsense' is set then to V
            else then verb annotation is omitted altogether.
        """
        self.mult = mult
        self.verbshandler = verbshandler
        self.rolehandler = rolehandler
        self.verbsonly = verbsonly
        self.deprel = deprel
        self.pairmap = pairmap

        print("DIRTAG initialized. " , end =" ")
        if self.verbshandler == 'complete':
            print(" Verbs are shown as is"  , end =" ")
        elif self.verbshandler == 'omitlemma':
            print(" Verb lemmas will be omitted." , end =" ")
        elif self.verbshandler == 'omitsense':
            print(" Verb senses will be omitted." , end =" ")
        elif self.verbshandler == 'omitverb' :
            print(" Verb senses are ignored." , end =" ")
        else :
            ParameterError()

        if self.rolehandler == 'complete':
            print(" Roles are shown as is")
        elif self.rolehandler == 'directionsonly':
            print(" Only directions will be shown for role tags.")
        elif self.rolehandler == 'rolesonly':
            print(" Directions will be omitted.")
        else :
            ParameterError()

        
 
    def encode(self, entry : conllu.CoNLL_U) -> List[str]:
        
        tags = [""] * len(entry)
        verblocs : List[int] = [ind for ind , val in enumerate(entry.get_vsa()) if val !="_"]
        vsa : List[str] = entry.get_vsa()
        
        annotations = [entry.get_srl_annotation(d) for d in range(entry.depth)]
        annT = [*zip(*annotations)]

        deprel = entry.get_by_tag("deprel")
        heads = [int(x)-1 for x in entry.get_by_tag("head")]
        depmask = [""] * len(entry)
        

        for row in range(len(annT)):
            for col in range(len(annT[row])):
                
                if annT[row][col] == "_":
                    continue
                
                if annT[row][col] == "V":
                    
                    if self.verbshandler == 'complete':
                        encoding = "V" + vsa[row]
                    elif self.verbshandler == 'omitlemma':
                        encoding = "V" + vsa[row][-2::]
                    elif self.verbshandler == 'omitsense':
                        encoding = "V"
                    else :
                        continue

                    if tags[row] == "":
                        tags[row] = encoding
                    
                    continue
                
                if self.verbsonly:
                    continue 

                # Remaning part is used to detect role words only.
                numdirsyms = -1 
                if row <= verblocs[col]:
                    symb =  ">"
                    f = int.__lt__
                else :
                    symb =  "<"
                    f = int.__gt__

                numdirsyms = len([1 for locs in verblocs if f(row,locs) and f(locs, verblocs[col])])+1
                if numdirsyms == 0:
                    Exception("Some Exception")
                
                if self.rolehandler == "complete":
                    encoding = symb * numdirsyms + annT[row][col]
                elif self.rolehandler =='rolesonly':
                    encoding = annT[row][col]
                else :
                    encoding = symb * numdirsyms 
        
                if tags[row] != "":
                    dirreg = re.compile("<*|>*")
                    match = dirreg.match(tags[row])
                    dif = match.end() - match.start() # We will replace new encoding with the old one if its pointing at a shorter distance.
                    if len(encoding) > dif:
                        # print(tags[row] , encoding)
                        continue

                tags[row] = encoding
        
        tagswithoutdir = [x.strip("<>") for x in tags]
        for index in range(len(tagswithoutdir)):
            if tagswithoutdir[index].startswith("V"):
                head = index
                while True:
                    head = heads[head]
                    if head == -1:
                        tags[index] = "*" + tags[index]
                        break
                    if head in verblocs:
                        hindex = verblocs.index(head)
                        if index in verblocs:
                            cindex = verblocs.index(index) 
                        else :
                            break

                        dif = hindex-cindex
                        if abs(dif) <= self.mult:
                            tags[index] = ("<"*(-dif) if dif < 0 else ">"*dif)+tags[index]
                        break

        if self.deprel:  
            for cur in range(len(entry)):
                if tags[cur] == "":
                    deptag = deprel[cur]
                    head = heads[cur]
                    for i in range(self.mult):
                        if head == -1:
                            # TODO 
                            break
                        if tags[head] != "" or vsa[head] != "_":

                            direction = 1 if cur < head else -1 
                            numofmarkers = 1 # designates how many '>'/'<' are necessary.
                            for i in range(cur , head , direction):
                                if i == cur :
                                    continue
                                if tags[i] != '':
                                    numofmarkers += 1
                            
                            if numofmarkers <= self.mult:
                                markers = numofmarkers*">" if direction == 1  else numofmarkers * "<"
                                depmask[cur] =  f"{markers}"

                            break
                        else :
                            head = heads[head]
                            deptag = deprel[head]
            
            for i in range(len(entry)):
                if tags[i] == "":
                    tags[i] = depmask[i]
                    


                    
        return tags    

    def decode(self , encoded : List[str] , vlocs : List[str]) -> List[List[str]] :
        
        verblocs = [i for i , v in enumerate(vlocs) if v != "" and v != "_"]
        numverb = len(verblocs)
        annotations = [["_" for i in range(len(encoded))] for i in range(numverb)]
        annlevelcounter = 0 
        
        for i , v in enumerate(vlocs):
            if v != "_" and v != "":
                annotations[annlevelcounter][i] = "V"
                annlevelcounter += 1
            

        if numverb == 0:
            return [["_" for i in range(len(encoded))]]


        for ind , val in enumerate(encoded):
            if val == "": 
                continue

            issrltagged= True if val.lstrip("<>") != "" else False

            if not issrltagged:
                continue


            pointsleft = False # Is the tag pointing to right '>' or left '<'?
            numpointers = 0 # How many '>'/'<' are there ?
            role = ""
            for j in val:
                if j == '>':
                    pointsleft = False
                    numpointers += 1
                elif j == '<':
                    pointsleft = True
                    numpointers += 1
                else:
                    role += j
            
            pointeddepth = -1
            if ind < verblocs[0] : pointeddepth = numpointers-1 
            elif  ind > verblocs[-1] : pointeddepth = len(verblocs) - numpointers
            else : 
                tempind = 0 
                for verbindex in verblocs:
                    if verbindex < ind:
                        tempind += 1
                    else:
                        pointeddepth = tempind + (numpointers - 1 if not pointsleft else -numpointers)
                        break
            
            
            if pointeddepth == -1 : 
                continue                
            
            
            try:
                annotations[pointeddepth][ind] = role
            except IndexError:
                continue 
            

            
        
        return annotations

    
    


