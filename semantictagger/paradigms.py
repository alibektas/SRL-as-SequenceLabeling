import abc

from numpy.core.numeric import outer
from pandas.core import frame
from semantictagger.reconstructor import ReconstructionModule
import numpy as np
from . import conllu
from typing import AsyncContextManager, Dict, Union , List , Tuple , NewType
import pdb 

from .edgedelegate import EdgeDelegate
from .selectiondelegate import SelectionDelegate
from .datatypes import * 
from .node import Node , ExtendedNode
from .edge import Edge
from .tagcandidate import TagCandidate
from semantictagger import edgedelegate
import enum 

EMPTY_LABEL = "_"
NUMPYGT = np.greater
NUMPYLT = np.less 

class RELPOSVERSIONS(enum.Enum):
    ORIGINAL = 0 
    SRLEXTENDED = 1
    SRLREPLACED = 2
    FLATTENED = 3
    DEPLESS = 4



class ParameterError(Exception):
    def __init__(self):
        super().__init__("False Parameter Error.")


class Encoder(abc.ABC):

    @abc.abstractclassmethod
    def encode(self,entry : conllu.CoNLL_U ) -> List[str]:
        pass

    @abc.abstractclassmethod
    def decode(self, encoded : List[str] , vlocs : List[str] = None) -> List[List[str]]:
        pass


    @abc.abstractclassmethod
    def spanize(self , entry : Union[conllu.CoNLL_U  , Tuple[List[str],List[str],List[str]]]) -> List[List[str]]:
        pass
        
    @abc.abstractclassmethod
    def to_conllu(self, words : List[str] , vlocs : List[str], encoded : List[str]):
        pass

    def test(self, entry : Union[conllu.CoNLL_U  , Tuple[conllu.CoNLL_U , List[str],List[str]]]) -> Tuple[int,int]:
        """
            Checks if an entry is correctly tagged by looking at
            whether it could be successfully converted back into 
            CoNLL_U.get_srl_annotation(self , depth) format.
        """
        if type(entry) == conllu.CoNLL_U:
            words = entry.get_words()
            vlocs = ["V" if x != "_" and x != "" else "_" for x in entry.get_vsa()]
            encoded = self.encode(entry)
            cmp = entry.get_span()

        else :
            words = entry[0].get_words()
            vlocs = entry[1]
            encoded = entry[2]
            cmp = entry[0].get_span()

        
        decoded = self.spanize(words  , vlocs , encoded)
        
        
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

class SRLPOS(Encoder):
   
    def __init__(
            self , 
            selectiondelegate : SelectionDelegate,
            reconstruction_module : ReconstructionModule,
            tag_dictionary : Dict[Tag,np.int],
            postype = POSTYPE.XPOS,
            frametype = FRAMETYPE.PREDONLY,
            version = RELPOSVERSIONS.SRLREPLACED
            ):
        """
            Semantically extended version of strzyz-et-al Viable Dependency Parsing
        """

        self.frametype = frametype
        self.postype = postype

        self.version = version

        self.roletagdict = tag_dictionary
        self.deptagdict = {}
        self.selectiondelegate = selectiondelegate
        self.reconstruction_module = reconstruction_module
    
        self.edgedelegate = EdgeDelegate()
        self.invdeptagdict = {self.deptagdict[i] : i  for i in self.deptagdict}
        self.invroletagdict = {self.roletagdict[i] : i  for i in self.roletagdict}
        
        print(f"SRLPOS() initalized.\n\tVersion:{self.version}\n\t{frametype}\n\t{postype}")


    
    def isdeplabel(self,label):
        return label.split(",")[1] != "LABEL"
        
    def resolvedeptag(self , index):
        return self.invdeptagdict[index]

    def resolveroletag(self, index):
        return self.invroletagdict[index]

    def resolvetag(self , tag :TagCandidate) -> Tag:
        return  ("-" if tag.direction == Direction.LEFT else "")+f"{tag.distance}",f"{self.resolveroletag(tag.tag)}" , tag.indexframe



    def maprole(self , tag : Tag):
        try: 
            return self.roletagdict[tag]
        except:
            index = len(self.roletagdict)
            self.roletagdict[tag] = index
            self.invroletagdict[index] = tag
        
        return index
    
    def mapdependency(self , tag : Tag):
        try: 
            return self.deptagdict[tag]
        except:
            index = len(self.deptagdict)
            self.deptagdict[tag] = index
            self.invdeptagdict[index] = tag
        
        return index

    def encode(self , entry:conllu.CoNLL_U) -> List[Tag]:

            

        tags = [""]*len(entry)
        pos = entry.get_by_tag("upos") if self.postype == POSTYPE.UPOS else entry.get_by_tag("xpos")
        deptags = entry.get_by_tag("deprel")
        verblocs = entry.get_verb_indices()
        heads = entry.get_heads()

        if  self.version == RELPOSVERSIONS.ORIGINAL:
            for i in range(len(entry)):
                head = heads[i]
                deptag = deptags[i]
                if head == -1 :
                    tags[i] = f"-1,ROOT,{deptag}"
                else :
                    headpos = pos[head]
                    direction = -1 if head < i else 1
                    numoccurence = 0 
                    for k in range(i + direction , head + direction, direction):
                        if headpos == pos[k]:
                            numoccurence += 1
                    tags[i] =  ("-" if direction == -1 else "") + f"{numoccurence},{headpos},{deptag}"
            return tags


        sentence = {i : Node(i , -1) if i == -1 else Node(i,heads[i]) for i in range(-1 ,len(entry))}
        
        self.edgedelegate.clear()
        self.edgedelegate.loadsentence(sentence)

        ROOT = sentence[-1]
        assert(ROOT.index == -1)

        if len(verblocs) != 0:
            # for i in verblocs: 
            #     edge  = Edge(i , ROOT.index ,Roletag(self.roletagdict["V"]) , Deptag(self.mapdependency(deptags[i])) , distance=-1 , direction=-1)
            #     self.edgedelegate.add(edge)


            for level , annotation in enumerate(entry.get_srl_annotation()):
                vindex = verblocs[level]
                for wordindex , role in enumerate(annotation):
                    if vindex == wordindex : continue 
                    if role != "_":
                        direction = Direction.RIGHT if wordindex < vindex else Direction.LEFT
                        cond , invcond =   (NUMPYLT, NUMPYGT) if direction == Direction.LEFT else (NUMPYGT, NUMPYLT)
                        distance = np.sum([1 if cond(index,wordindex) and invcond(index,vindex) else 0 for index in verblocs]) + 1 
                        edge = Edge(wordindex , vindex , Roletag(self.maprole(role)) , Deptag(self.mapdependency("_")) , distance , direction)
                        self.edgedelegate.add(edge)

            
            # Offer tag candidates for entry's words.
            for i in verblocs:
                verb = sentence[i]
                for edgeid in verb.incoming:
                    edge : Edge = self.edgedelegate.get(edgeid)
                    tagcandidate = TagCandidate( edge.direction , edge.distance , edge.roletag , 1 , i)
                    sentence[edge.from_].addtag(tagcandidate)
                    
        
        for i in range(len(entry)):
            word = sentence[i]
            head = heads[i]
            deptag = deptags[i]
            if word.isheadword():
                roledirection , roletag , indexframe = self.resolvetag(self.selectiondelegate.select(word.tags))
                if self.version == RELPOSVERSIONS.SRLREPLACED:
                    tags[i] = f"{roledirection},FRAME,{roletag}"
                elif self.version == RELPOSVERSIONS.SRLEXTENDED:
                    if head == -1:
                        tags[i] = f"-1,ROOT,root,{roledirection},{roletag}"
                        continue
                    else :
                        headpos = pos[head]
                    direction = -1 if head < i else 1
                    numoccurence = 0 
                    for k in range(i + direction , head + direction, direction):
                        if headpos == pos[k]:
                            numoccurence += 1
                    tags[i] =  ("-" if direction == -1 else "") + f"{numoccurence},{headpos},{deptag},{roledirection},{roletag}"
                elif self.version == RELPOSVERSIONS.FLATTENED:
                    head = indexframe
                    headpos = pos[head]
                    direction = -1 if head < i else 1
                    numoccurence = 0 
                    for k in range(i + direction , head + direction, direction):
                        if headpos == pos[k]:
                            numoccurence += 1
                    tags[i] =  ("-" if direction == -1 else "") + f"{numoccurence},{headpos},{roletag}"  
                else:
                    head = indexframe
                    headpos = pos[head]
                    direction = -1 if head < i else 1
                    numoccurence = 0 
                    for k in range(i + direction , head + direction, direction):
                        if headpos == pos[k]:
                            numoccurence += 1
                    tags[i] =  ("-" if direction == -1 else "") + f"{numoccurence},{headpos},{roletag}"


            else:   
                if self.version == RELPOSVERSIONS.SRLREPLACED or self.version == RELPOSVERSIONS.SRLEXTENDED:
                    if head == -1 :
                        tags[i] = f"-1,ROOT,{deptag}"
                    else :
                        headpos = pos[head]
                        direction = -1 if head < i else 1
                        numoccurence = 0 
                        for k in range(i + direction , head + direction, direction):
                            if headpos == pos[k]:
                                numoccurence += 1
                        tags[i] =  ("-" if direction == -1 else "") + f"{numoccurence},{headpos},{deptag}"
                elif self.version == RELPOSVERSIONS.DEPLESS:
                    if head == -1 :
                        tags[i] = f"-1,ROOT"
                    else :
                        headpos = pos[head]
                        direction = -1 if head < i else 1
                        numoccurence = 0 
                        for k in range(i + direction , head + direction, direction):
                            if headpos == pos[k]:
                                numoccurence += 1
                        tags[i] =  ("-" if direction == -1 else "") + f"{numoccurence},{headpos}"
                elif self.version == RELPOSVERSIONS.FLATTENED:
                    if head == -1 :
                        tags[i] = f"-1,ROOT"
                    else :
                        while not sentence[head].isheadword() and head != -1:
                            head = heads[head]
                        if head == -1:
                            tags[i] = f"-1,ROOT"
                        
                        headpos = pos[head]
                        direction = -1 if head < i else 1
                        numoccurence = 0 
                        for k in range(i + direction , head + direction, direction):
                            if headpos == pos[k]:
                                numoccurence += 1
                        tags[i] =  ("-" if direction == -1 else "") + f"{numoccurence},{headpos}"
                


        return tags 



    def decode(self , encoded : List[Tag] , vlocs : List[Tag], postags : List[Tag]) -> Annotation :
        
        if self.version == RELPOSVERSIONS.ORIGINAL:
            NotImplementedError()

        verblocs = [i for i , v in enumerate(vlocs) if v != "_"]
        numverb = len(verblocs)
        annotations : Annotation = [["_" for i in range(len(encoded))] for i in range(numverb)]
        annlevelcounter = 0 
        
        for i , v in enumerate(vlocs):
            if v != "_" and v != "":
                annotations[annlevelcounter][i] = "V"
                annlevelcounter += 1
            

        if numverb == 0:
            return [["_" for i in range(len(encoded))]]


        for ind , val in enumerate(encoded):
            
            if val == "" or val == "<UNKNOWN>": 
                continue
            
            temp = val.split(",")
            
            if self.version == RELPOSVERSIONS.SRLEXTENDED:
                if len(temp) <= 3:
                    continue
                distance , roledeptag = int(temp[3]) , temp[4]

            elif self.version == RELPOSVERSIONS.SRLREPLACED:
                if len(temp) < 3:
                    continue
                distance , possrl , roledeptag = int(temp[0]) , temp[1] , temp[2]
                issrltagged = True if possrl == "FRAME" else False
                if not issrltagged:
                    continue

            elif self.version == RELPOSVERSIONS.FLATTENED:
                if len(temp) == 3:
                    distance , possrl , roledeptag = int(temp[0]) , temp[1] , temp[2]
                else :
                    continue
            
            elif self.version == RELPOSVERSIONS.DEPLESS:
                if len(temp) == 3:
                    distance , possrl , roledeptag = int(temp[0]) , temp[1] , temp[2]
                else :
                    continue




            pointsleft = True if distance < 0 else False
            numpointers = np.abs(distance)
            role = roledeptag
            pointeddepth = -1

            if self.version == RELPOSVERSIONS.SRLEXTENDED or self.version == RELPOSVERSIONS.SRLREPLACED:
                if ind < verblocs[0] : pointeddepth = numpointers-1 
                elif ind == verblocs[0] : pointeddepth = numpointers
                elif  ind > verblocs[-1] : pointeddepth = len(verblocs) - numpointers
                elif ind == verblocs[-1] : pointeddepth = len(verblocs) - numpointers - 1
                else : 
                    tempind = 0 
                    for verbindex in verblocs:
                        if verbindex < ind:
                            tempind += 1
                        elif verbindex == ind:
                            pointeddepth = tempind + (numpointers  if not pointsleft else -numpointers)
                            break
                        else:
                            pointeddepth = tempind + (numpointers - 1 if not pointsleft else -numpointers)
                            break
                
            else :
                foundheads = 0
                for index  in range(ind + ( 1 if not pointsleft else -1) , len(encoded) if not pointsleft else -1 , 1 if not pointsleft else -1):
                    if possrl == postags[index]:
                        foundheads += 1
                        if foundheads == numpointers:
                            try:
                                pointeddepth = verblocs.index(index)
                            except ValueError:
                                pointeddepth = 0 
                            break
                roletag = False

            if pointeddepth < 0 : 
                IndexError("Pointed depth cant be less than 0.")        
                  
            
            try:
                annotations[pointeddepth][ind] = role
            except IndexError:
                continue 
            
        return annotations

    
    def spanize(self , words : List[str] , vlocs : List[Tag], encoded : List[Tag] , pos : List[Tag]) -> Annotation:
        if self.version == RELPOSVERSIONS.ORIGINAL:
            NotImplementedError()

        entry : conllu.CoNLL_U = self.to_conllu(words , vlocs , encoded , pos)
        return self.reconstruct(entry)

    def reconstruct(self , entry : conllu.CoNLL_U):
        self.reconstruction_module.loadsentence(entry)
        return self.reconstruction_module.reconstruct()

    def to_conllu(self, words: List[str], vlocs: List[str], encoded: List[str] , pos : List[str]):

        if self.version == RELPOSVERSIONS.ORIGINAL:
            NotImplementedError()
        
        decoded = self.decode(encoded , vlocs , pos)

        encoded = [tuple(x.split(",")) for x in encoded]
        dT = [*zip(*decoded)]

        content = []
        roletag = False

        for j in range(len(words)):
            
            if encoded[j][0] == "<UNKNOWN>" or len(encoded[j])==0:
                posfields = ["_" , pos[j]] if self.postype == POSTYPE.XPOS else ["_" , pos[j]] 

                dict_ = {
                "form" : words[j] ,
                "lemma" : "_" ,
                "upos" : posfields[0],
                "xpos" : posfields[1],
                "feats" : "_",
                "head" : str(0),
                "deprel" : "_",
                "vsa": vlocs[j],
                "srl" : list(dT[j])
                } 
            
                content.append(dict_)
                continue


            if self.version == RELPOSVERSIONS.SRLEXTENDED:
                if len(encoded[j]) == 3:
                    distance , postag , roledeptag = encoded[j]
                    srl = None
                else :
                    distance , postag , roledeptag , _ , srl = encoded[j]
            elif self.version == RELPOSVERSIONS.SRLREPLACED:
                distance , postag , deptag = encoded[j]
            elif self.version == RELPOSVERSIONS.FLATTENED or self.version == RELPOSVERSIONS.DEPLESS:
                if len(encoded[j]) == 3 :
                    distance , postag , _ = encoded[j]
                elif len(encoded[j])== 2:
                    distance , postag = encoded[j]

                

            
            distance = int(distance)
            numofmarkers = abs(distance)
            pointsleft = True if distance < 0 else False
            foundheads = 0 
            head = -1 

            if (self.version == RELPOSVERSIONS.SRLREPLACED and postag != "FRAME") or (self.version == RELPOSVERSIONS.SRLEXTENDED) or (self.version == RELPOSVERSIONS.FLATTENED) or self.version == RELPOSVERSIONS.DEPLESS:
                
                for index  in range(j + ( 1 if not pointsleft else -1) , len(encoded) if not pointsleft else -1 , 1 if not pointsleft else -1):
                    if postag == pos[index]:
                        foundheads += 1
                        if foundheads == numofmarkers:
                            head = index
                            break
                roletag = False
            else:
                for index  in range(j + ( 1 if not pointsleft else -1) , len(encoded) if not pointsleft else -1 , 1 if not pointsleft else -1):
                    if vlocs[index] != "_":
                        foundheads += 1
                        if foundheads == numofmarkers:
                            head = index
                            break
                roletag = True

            posfields = ["_" , pos[j]] if self.postype == POSTYPE.XPOS else ["_" , pos[j]] 


            rolelabel = "_"
            if self.version == RELPOSVERSIONS.SRLREPLACED :
                rolelabel = deptag if not roletag else "_" 
            elif self.version == RELPOSVERSIONS.SRLEXTENDED:
                rolelabel = roledeptag
            else :
                # For the remaining two versions deprel is not used.
                pass

            dict_ = {
                "form" : words[j] ,
                "lemma" : "_" ,
                "upos" : posfields[0],
                "xpos" : posfields[1],
                "feats" : "_",
                "head" : str(head + 1),
                "deprel" : rolelabel,
                "vsa": vlocs[j],
                "srl" : list(dT[j])
            } 
            


            content.append(dict_)
        
        entry = conllu.CoNLL_U(content)
        return entry