import abc
from . import conllu , tag 
from typing import List , Tuple
import re


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

    def __init__(self , mult  , verbshandler : str = 'complete' , verbsonly = False):
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
        self.verbsonly = verbsonly

        print("Active tagger DIRTAG " , end =" ")
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

        
    

    def encode(self, entry : conllu.CoNLL_U) -> List[str]:
        
        tags = [""] * len(entry)
        verblocs : List[int] = [ind for ind , val in enumerate(entry.get_vsa()) if val !="_"]
        vsa : List[str] = entry.get_vsa()
        
        annotations = [entry.get_srl_annotation(d) for d in range(entry.depth)]
        annT = [*zip(*annotations)]

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

                encoding = symb * numdirsyms + annT[row][col] 
        
                if tags[row] != "":
                    if tags[row].startswith("V"):
                        continue

                tags[row] = encoding
                    
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

