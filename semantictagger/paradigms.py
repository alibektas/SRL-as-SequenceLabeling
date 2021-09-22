import abc

from numpy.core.numeric import outer
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


EMPTY_LABEL = "_"
NUMPYGT = np.greater
NUMPYLT = np.less 


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

class DIRECTTAG(Encoder):
    """
        Directional Encoder. Every word is responsible for holding its 
        own tag. To show that a non-predicate is connected to a predicate,  
        '>' or '<' , depending on predicates relative location , are used.
        '>' simply means next predicate to the right.
        
        Multiple uses are allowed  : e.g '>>ARG0'
        This multiplicity can be limited by mult arg of this class. 
    """

    def __init__(
            self , 
            mult,
            selectiondelegate : SelectionDelegate,
            reconstruction_module : ReconstructionModule,
            tag_dictionary : Dict[Tag,np.int],
            rolehandler : str = 'complete',
            verbshandler : str = 'complete',
            verbsonly = False,
            deprel = False
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

        self.roletagdict = tag_dictionary
        self.deptagdict = {}
        self.selectiondelegate = selectiondelegate
        self.reconstruction_module = reconstruction_module
        
    
        self.edgedelegate = EdgeDelegate()

        assert(mult>0)
        counter = 0 
        for i in range(-mult,mult):
            if i < 0 :
                self.deptagdict["<"*(-mult)] = counter
            elif i == 0 :
                self.deptagdict["_"] = counter
                self.deptagdict[""] = counter
            else :
                self.deptagdict[">"*mult] = counter
            counter += 1

        self.invdeptagdict = {self.deptagdict[i] : i  for i in self.deptagdict}
        self.invroletagdict = {self.roletagdict[i] : i  for i in self.roletagdict}


        print("DIRTAG initialized. " , end =" ")
        if self.verbshandler == 'complete':
            print(" Verbs are shown as is"  , end =" ")
        elif self.verbshandler == 'omitlemma':
            print(" Verb lemmas will be omitted." , end =" ")
        elif self.verbshandler == 'omitsense':
            print(" Verb senses will be omitted." , end =" ")
        elif self.verbshandler == 'omitverb' :
            print(" Verb senses are ignored." , end =" ")
        elif self.verbshandler == 'orderverbs':
            print(" Verbs will be order in terms of dependent-governor relation." , end =" ")
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

    
    def isdeplabel(self,label):
        a = label.replace("<", "").replace(">","")
        return a == ""

    def resolvedeptag(self , index):
        return self.invdeptagdict[index]

    def resolveroletag(self, index):
        return self.invroletagdict[index]

    def resolvetag(self , tag :TagCandidate) -> Tag:
        return ("<" if tag.direction == Direction.LEFT else ">") * tag.distance + self.resolveroletag(tag.tag)



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
        verblocs = entry.get_verb_indices()
        heads = entry.get_heads()
        sentence = {i : Node(i , -1) if i == -1 else Node(i,heads[i]) for i in range(-1 ,len(entry))}
        
        self.edgedelegate.clear()
        self.edgedelegate.loadsentence(sentence)

        ROOT = sentence[-1]
        assert(ROOT.index == -1)

        if len(verblocs) == 0:
            return ["_"]*len(entry)

        for i in verblocs: 
            edge  = Edge(i , ROOT.index ,Roletag(self.roletagdict["V"]) , Deptag(self.deptagdict["_"]) , distance=-1 , direction=-1)
            self.edgedelegate.add(edge)


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
                tagcandidate = TagCandidate( edge.direction , edge.distance , edge.roletag , 1)
                sentence[edge.from_].addtag(tagcandidate)
                
        for i in range(len(entry)):
            word = sentence[i]
            isverb = i in verblocs

            if word.isheadword():
                tags[i] = self.resolvetag(self.selectiondelegate.select(word.tags))
            else:
                if isverb:
                    head = word.index
                    headword = sentence[head]
                    while True:
                        head = sentence[head].head
                        headword = sentence[head]
                        if head == -1:
                            outerloopcontinues = False
                            break

                        pointing = []
                        for og in headword.outgoing:
                            edge = self.edgedelegate.get(og)
                            if edge.to_ == i:
                                break
                            else :
                                pointing.append(edge.to_)
                        else :
                            if len(pointing) == 0:
                                continue
                            
                            if len(pointing) == 1 and pointing[0] == -1:
                                outerloopcontinues = False
                                break

                            newverborder = np.argmax(pointing)
                            newverborder = np.argmax(verblocs == pointing)
                            oldverborder = np.argmax(verblocs == i)
                            distance = np.abs(newverborder-oldverborder)
                            
                            if distance == 0 : 
                                outerloopcontinues = False
                                break

                            direction = Direction.LEFT if newverborder < oldverborder else Direction.RIGHT
                            outerloopcontinues = True
                            tags[i] = ("<" if direction == Direction.LEFT else ">")* distance + "+"
                            break
                    
                    if outerloopcontinues:
                        continue
                    
                head = word.head
                while True:
                    
                    if head == -1 : 
                        break
                    if not sentence[head].isheadword() and not head in verblocs:
                        head = sentence[head].head
                        continue
                        
                    direction = 1 if  i < head else -1
                    nummarkers = 1
                    for j in range(i+direction , head , direction):
                        if sentence[j].isheadword() or j in verblocs:
                            nummarkers += 1 
                    tags[i] = ("<" if direction == -1 else ">")* nummarkers
                    break

        return tags 



    def decode(self , encoded : List[Tag] , vlocs : List[Tag]) -> Annotation :
        
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
            # if role == "+":
            #     continue 
            
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
            
            if pointeddepth == -1 : 
                continue                
            
            try:
                annotations[pointeddepth][ind] = role
            except IndexError:
                continue 
            
        return annotations

    
    def spanize(self , words : List[str] , vlocs : List[Tag], encoded : List[Tag]) -> Annotation:

        entry : conllu.CoNLL_U = self.to_conllu(words , vlocs , encoded)
        self.reconstruction_module.loadsentence(entry)
        return self.reconstruction_module.reconstruct()
        

    def to_conllu(self, words: List[str], vlocs: List[str], encoded: List[str]):

        decoded = self.decode(encoded , vlocs)
        dT = [*zip(*decoded)]

        content = []


        for j in range(len(words)):

            head = -1 
            if encoded[j] == "":
                # TODO : what's a better value for this?
                head = -1
            else:
                if self.isdeplabel(encoded[j]):
                    numofmarkers = len(encoded[j])
                    pointsleft = True if encoded[j][0] == "<" else False
                    foundverbs = 0 
                    for index  in range(j + ( 1 if not pointsleft else -1) , len(encoded) if not pointsleft else -1 , 1 if not pointsleft else -1):
                        if vlocs[index] != "_" or not self.isdeplabel(encoded[index]):
                            foundverbs += 1
                            if foundverbs == numofmarkers:
                                head = index
                                break
                else:
                    numofmarkers = 0
                    for c in encoded[j]:
                        if c == "<" or c == ">":
                            numofmarkers += 1
                        else :
                            break

                    pointsleft = True if encoded[j][0] == "<" else False
                    foundverbs = 0
                    firstocc = False
                    
                    for i in range(j , -1 if pointsleft else len(vlocs) , -1 if pointsleft else 1):
                        if i == j :
                            continue 
                        if vlocs[i] != "_":
                            if not firstocc : 
                                firstocc = i
                            numofmarkers -= 1
                            if numofmarkers == 0:
                                head = i 
                                break
                    else :
                        if not firstocc: head = -1
                        else : head = firstocc 
        

            dict_ = {
                "form" : words[j] ,
                "lemma" : "_" ,
                "upos" : "_" ,
                "xpos" : "_",
                "feats" : "_",
                "head" : str(head + 1),
                "deprel" : "_",
                "vsa": vlocs[j],
                "srl" : list(dT[j])
            } 

            content.append(dict_)
        
        entry = conllu.CoNLL_U(content)
        return entry


class SEQTAG(Encoder):
   
    def __init__(
            self , 
            mult,
            selectiondelegate : SelectionDelegate,
            reconstruction_module : ReconstructionModule,
            tag_dictionary : Dict[Tag,np.int],
            rolehandler : str = 'complete',
            verbshandler : str = 'complete',
            verbsonly = False,
            deprel = False
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

        self.roletagdict = tag_dictionary
        self.deptagdict = {}
        self.selectiondelegate = selectiondelegate
        self.reconstruction_module = reconstruction_module
        
    
        self.edgedelegate = EdgeDelegate()

        assert(mult>0)
        counter = 0 
        for i in range(-mult,mult):
            if i < 0 :
                self.deptagdict["<"*(-mult)] = counter
            elif i == 0 :
                self.deptagdict["_"] = counter
                self.deptagdict[""] = counter
            else :
                self.deptagdict[">"*mult] = counter
            counter += 1

        self.invdeptagdict = {self.deptagdict[i] : i  for i in self.deptagdict}
        self.invroletagdict = {self.roletagdict[i] : i  for i in self.roletagdict}


        print("DIRTAG initialized. " , end =" ")
        if self.verbshandler == 'complete':
            print(" Verbs are shown as is"  , end =" ")
        elif self.verbshandler == 'omitlemma':
            print(" Verb lemmas will be omitted." , end =" ")
        elif self.verbshandler == 'omitsense':
            print(" Verb senses will be omitted." , end =" ")
        elif self.verbshandler == 'omitverb' :
            print(" Verb senses are ignored." , end =" ")
        elif self.verbshandler == 'orderverbs':
            print(" Verbs will be order in terms of dependent-governor relation." , end =" ")
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

    
    def isdeplabel(self,label):
        return label.split(",")[1] != "LABEL"
        
    def resolvedeptag(self , index):
        return self.invdeptagdict[index]

    def resolveroletag(self, index):
        return self.invroletagdict[index]

    def resolvetag(self , tag :TagCandidate) -> Tag:
        return  ("-" if tag.direction == Direction.LEFT else "") + f"{tag.distance},FRAME,{self.resolveroletag(tag.tag)}"



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
        pos = entry.get_pos()
        deptags = entry.get_by_tag("deprel")
        verblocs = entry.get_verb_indices()
        heads = entry.get_heads()

        sentence = {i : Node(i , -1) if i == -1 else Node(i,heads[i]) for i in range(-1 ,len(entry))}
        
        self.edgedelegate.clear()
        self.edgedelegate.loadsentence(sentence)

        ROOT = sentence[-1]
        assert(ROOT.index == -1)

        if len(verblocs) != 0:
            for i in verblocs: 
                edge  = Edge(i , ROOT.index ,Roletag(self.roletagdict["V"]) , Deptag(self.deptagdict["_"]) , distance=-1 , direction=-1)
                self.edgedelegate.add(edge)


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
                    tagcandidate = TagCandidate( edge.direction , edge.distance , edge.roletag , 1)
                    sentence[edge.from_].addtag(tagcandidate)
                    
        
        for i in range(len(entry)):
            word = sentence[i]

            if word.isheadword():
                tags[i] = self.resolvetag(self.selectiondelegate.select(word.tags))
            else:
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



    def decode(self , encoded : List[Tag] , vlocs : List[Tag]) -> Annotation :
        
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
            
            if val == "": 
                continue
            
            temp = val.split(",")
            distance , possrl , roledeptag = int(temp[0]) , temp[1] , temp[2]

            issrltagged = True if possrl == "FRAME" else False
            if not issrltagged:
                continue


            pointsleft = True if distance < 0 else False
            numpointers = np.abs(distance)
            role = roledeptag
            pointeddepth = -1

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
            
            if pointeddepth == -1 : 
                continue                
            
            try:
                annotations[pointeddepth][ind] = role
            except IndexError:
                continue 
            
        return annotations

    
    def spanize(self , words : List[str] , vlocs : List[Tag], encoded : List[Tag] , pos : List[Tag]) -> Annotation:

        entry : conllu.CoNLL_U = self.to_conllu(words , vlocs , encoded , pos)
        self.reconstruction_module.loadsentence(entry)
        return self.reconstruction_module.reconstruct()
        

    def to_conllu(self, words: List[str], vlocs: List[str], encoded: List[str] , pos : List[str]):
        decoded = self.decode(encoded , vlocs)
        encoded = [tuple(x.split(",")) for x in encoded]
        dT = [*zip(*decoded)]

        content = []


        for j in range(len(words)):

            distance , possrl , roledeptag = encoded[j]
            distance = int(distance)
            numofmarkers = abs(distance)
            pointsleft = True if distance < 0 else False
            foundheads = 0 
            head = -1 
            
            if possrl != "FRAME":
                for index  in range(j + ( 1 if not pointsleft else -1) , len(encoded) if not pointsleft else -1 , 1 if not pointsleft else -1):
                    if possrl == pos[index]:
                        foundheads += 1
                        if foundheads == numofmarkers:
                            head = index
                            break
            else:
                for index  in range(j + ( 1 if not pointsleft else -1) , len(encoded) if not pointsleft else -1 , 1 if not pointsleft else -1):
                    if vlocs[index] != "_":
                        foundheads += 1
                        if foundheads == numofmarkers:
                            head = index
                            break
                
            dict_ = {
                "form" : words[j] ,
                "lemma" : "_" ,
                "upos" : "_" ,
                "xpos" : "_",
                "feats" : "_",
                "head" : str(head + 1),
                "deprel" : "_",
                "vsa": vlocs[j],
                "srl" : list(dT[j])
            } 

            content.append(dict_)
        
        entry = conllu.CoNLL_U(content)
        return entry