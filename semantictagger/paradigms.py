import abc
import re

from . import conllu
from . import dataset 
from typing import Dict, Mapping , Union , List , Tuple 

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

        
class Encoder(abc.ABC):
    
    @abc.abstractclassmethod
    def encode(self,entry : conllu.CoNLL_U) -> List[str]:
        pass

    @abc.abstractclassmethod
    def decode(self,encoded : List[str]) -> List[List[str]]:
        pass

    def test(self, entry : conllu.CoNLL_U) -> Tuple[int,int]:
        """
            Checks if an entry is correctly tagged by looking at
            whether it could be successfully converted back into 
            CoNLL_U.get_srl_annotation(self , depth) format.
        """
        
        encoded = self.encode(entry)
        decoded = self.decode(encoded)

        cmp = [entry.get_srl_annotation(d) for d in range(entry.depth)]
        
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
            return correct , false



        return correct , false

class BNE(Encoder):
    def __init__(self , assign_limit , can_distribute = True , assigned_can_delegate = True):
        self.assign_limit = assign_limit
        self.can_distribute = can_distribute
        self.assigned_can_delegate = assigned_can_delegate
        

    def encode(self,entry):

        # Not white-space or Verb.
        annotations = [entry.get_srl_annotation(d) for d in range(entry.depth)]
        vsa = entry.get_vsa()
        
        counter = 1 
        indexing = {}

        T_ann = [*zip(*annotations)]
        tags = [tag.Tag() for i in range(len(entry.get_words()))]


        for col in range(len(T_ann)):
            for row in range(len(T_ann[col])):
                if not(T_ann[col][row] == "V" or T_ann[col][row]== tag.Tag.EMPTYTAG):
                    if col not in indexing:
                        indexing[col] = counter
                        tags[col].assigned = counter
                        counter += 1
                        if counter == self.assign_limit :
                            counter = 1 

        for row in range(len(annotations)):
            encoding = []
            index_of_verb = -1
            for col in range(len(annotations[row])):
                if annotations[row][col] == "V":
                    index_of_verb = col
                    tags[index_of_verb].vsa =  vsa[index_of_verb] 
                elif annotations[row][col] == tag.Tag.EMPTYTAG:
                    continue
                else :
                    encoding.append((indexing[col] , annotations[row][col]))
            if index_of_verb == -1 and encoding:
                print("No verb found in this round please check")
            tags[index_of_verb].delegates += encoding


        distribute_tag_index = -1
        last_available_index = -1
        distribution_active = False

        if self.can_distribute:
            for index in range(len(tags)):
                if tags[index].isverb():
                    if not distribution_active:
                        distribute_tag_index = index
                        last_available_index = index
                        distribution_active = True
                    else :
                        if distribute_tag_index != last_available_index:
                            while tags[distribute_tag_index].delegates:
                                tags[last_available_index].delegates.append(tags[distribute_tag_index].delegates.pop(0))
                        distribute_tag_index = index
                        last_available_index = index
                elif tags[index].isassigned():
                    if distribution_active:
                        if self.assigned_can_delegate:
                            last_available_index = index
                            if tags[distribute_tag_index].delegates :
                                tags[last_available_index].delegates.append(tags[distribute_tag_index].delegates.pop(0))
                            else :
                                distribution_active = False
                        else:
                            continue
                else :
                    if distribution_active:
                        last_available_index = index
                        if tags[distribute_tag_index].delegates :
                            tags[last_available_index].delegates.append(tags[distribute_tag_index].delegates.pop(0))
                        else :
                            distribution_active = False

        return [str(tag) for tag in tags]
      
    
    def decode(self,encoded):
        
        size = len(encoded)
        numverb = 0
        assigned_indices = [{}]
        indexlevel = 0

        for i in range(size):
            if encoded[i] == tag.Tag.EMPTYTAG:
                continue
            else :
                
                if encoded[i].startswith("V"): 
                    if "*" in encoded[i]:
                        vsa , assigned = encoded[i].split("++")[0].split("*")
                        if assigned in assigned_indices[indexlevel]:
                            indexlevel +=1
                            assigned_indices.append({})
                        assigned_indices[indexlevel][assigned] = i 
                    numverb += 1
                else:
                    if encoded[i].startswith("++"):
                        continue
                    else:
                        assigned = encoded[i].split("++")[0]
                        if assigned in assigned_indices[indexlevel]:
                            indexlevel +=1
                            assigned_indices.append({})
                        assigned_indices[indexlevel][assigned] = i 
                        
        
        annotations = [] 

        if numverb == 0 :
            return [['_'] * size]

        for i in range(numverb):
            annotations.append([tag.Tag.EMPTYTAG] * size)
        
        indexlevel = 0 
        depth = -1
        seen_indices = []
        verblevel = 0

        for i in range(size):
            if encoded[i] != tag.Tag.EMPTYTAG:
                startindex = 0 

                # Get all the tags and assigned number etc.
                if encoded[i].startswith("++"):
                    arr = encoded[i].split("++")[1::]
                else :
                    arr = encoded[i].split("++")


                if arr[0].startswith("V"):
                    depth += 1
                    startindex = 1 
                    
                    annotations[verblevel][i] = 'V'
                    verblevel += 1

                    if "*" in arr[0]:
                        startindex = 1
                        firstelem = arr[0].split("*")
                        vsa , assigned_index = firstelem[0] , firstelem[1]
                        if assigned_index in seen_indices:
                            indexlevel += 1
                            seen_indices = [assigned_index]
                        else :
                            if int(assigned_index) ==  self.assign_limit:
                                seen_indices = []
                                indexlevel += 1
                            else:
                                seen_indices.append(assigned_index)
                elif not "," in arr[0]:
                    startindex = 1
                    if arr[0] in seen_indices:
                        indexlevel += 1
                        seen_indices = [arr[0]]
                    else:
                        if int(arr[0]) ==  self.assign_limit:
                            seen_indices = []
                            indexlevel +=1
                        else :
                            seen_indices.append(arr[0])
                        
                for j in range(startindex,len(arr)):
                    tag = arr[j]
                    reference , role = tag.split(",")
                    resolvedindex = assigned_indices[indexlevel][reference]
                    annotations[depth][resolvedindex] = role
            else :
                continue

        return annotations

    
    
class GLOB(Encoder):
    def __init__(self , add_verb_lemma = False , abs= True):
        self.add_verb_lemma = add_verb_lemma
        self.abs = abs

    def encode(self , entry) :
        annotations = [entry.get_srl_annotation(d) for d in range(entry.depth)]
        tags = [""] * len(entry.get_words())
        vsa = entry.get_vsa()
        
        for row in range(len(annotations)):
            index_of_verb = -1
            encodings = []
            for col in range(len(annotations[row])):
                if annotations[row][col] == "V":
                    index_of_verb = col
                    if self.add_verb_lemma:
                        tags[col] = vsa[col]
                    else :
                        tags[col] = vsa[col][-2::]
                elif  annotations[row][col] == "_":
                    continue
                else :
                    encodings.append((col,annotations[row][col]))
            
            for index , tag in encodings:
                if abs:
                    tags[index_of_verb] += "++" + str(index) + tag
                else : 
                    tags[index_of_verb] += "++" + str(index - index_of_verb) + tag
            
        for index in range(len(tags)):
                if tags[index] == "":
                    tags[index] = tag.Tag.EMPTYTAG

        return tags

    def decode(self , encoded):
        NotImplementedError()

        
class LOCTAG(Encoder):
    def encode(self,entry):
        NotImplementedError()

    def decode(self,encoded):
        NotImplementedError()


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
            depreldepth = 1
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
        self.depreldepth = depreldepth

        print("DIRTAG initialized. " , end =" ")
        if self.verbshandler == 'complete':
            print("Verbs are shown as is")
        elif self.verbshandler == 'omitlemma':
            print("Verb lemmas will be omitted.")
        elif self.verbshandler == 'omitsense':
            print("Verb senses will be omitted.")
        elif self.verbshandler == 'omitverb' :
            print("Verb senses are ignored.")
        else :
            ParameterError()

        if self.rolehandler == 'complete':
            print("Roles are shown as is")
        elif self.rolehandler == 'directionsonly':
            print("Only directions will be shown for role tags.")
        elif self.rolehandler == 'rolesonly':
            print("Directions will be omitted.")
        else :
            ParameterError()

        
 
    def encode(self, entry : conllu.CoNLL_U) -> List[str]:
        
        tags = [""] * len(entry)
        verblocs : List[int] = [ind for ind , val in enumerate(entry.get_vsa()) if val !="_"]
        vsa : List[str] = entry.get_vsa()
        
        annotations = [entry.get_srl_annotation(d) for d in range(entry.depth)]
        annT = [*zip(*annotations)]

        if self.deprel:
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
                        encoding = ""

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

                numdirsyms = min(self.mult , len([1 for locs in verblocs if f(row,locs) and f(locs, verblocs[col])])+1)
                if numdirsyms == 0:
                    Exception("Some Exception")
                
                if self.rolehandler == "complete":
                    encoding = symb * numdirsyms + annT[row][col]
                elif self.rolehandler =='rolesonly':
                    encoding = annT[row][col]
                else :
                    encoding = symb * numdirsyms 
        
                if tags[row] != "":
                    if tags[row].startswith("V"):
                        continue

                tags[row] = encoding
        
        if self.deprel:  
            for cur in range(len(entry)):
                if tags[cur] == "":
                    deptag = deprel[cur]
                    head = heads[cur]
                    for i in range(self.depreldepth):
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
                            if numofmarkers > self.mult:
                                break
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

    def decode(self , encoded : List[str]) -> List[List[str]] :
        
        verblocs = [ind for ind,val in enumerate(encoded) if val.startswith("V")]
        numverb = len(verblocs)
        annotations = [["_" for i in range(len(encoded))] for i in range(numverb)]
        levelverb = 0

        if numverb == 0:
            return [["_" for i in range(len(encoded))]]

        for ind ,val in enumerate(encoded):
            if val == "": 
                continue
            if val.startswith("V"):
                annotations[verblocs.index(ind)][ind] = "V"
            else:
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
                    for verbindex  in range(len(verblocs)-1):
                        if verblocs[verbindex] <= ind and ind < verblocs[verbindex+1] :
                            pointeddepth = verbindex + (-1 if pointsleft else 1 ) * (numpointers-1)
                            break
                
                
                if pointeddepth == -1 : 
                    print("No depth found.")
                    pass
                
                
                annotations[pointeddepth][ind] = role
                
        
        return annotations

    
class RELPOS(Encoder):
    def __init__(self, mapping : Mapper = None):
        self.mapping = mapping


    def encode(self , sentence : conllu.CoNLL_U):
        encoded : List[str] = [''] * len(sentence)

        heads = [int(i)-1 for i in sentence.get_by_tag("head")]
        deprels = sentence.get_by_tag("deprel")
        pos = sentence.get_by_tag("upos")

        for index in range(len(sentence)):
            head = heads[index]
            dep = deprels[index]
            postag = pos[head]
            dir = 1 if index < head else -1
            offset = dir 
            for tmp in range(index , head , dir):
                if tmp == index : continue 
                if postag == pos[tmp]:
                    offset += dir 
            if self.mapping is not None:
                encoded[index]= self.mapping.maptag(f"{postag}|{offset}|{dep}")
            else:
                encoded[index] = f"{postag}|{offset}|{dep}"
            
        return encoded 

    def decode(self, encoded : List[str]):
        NotImplementedError()



