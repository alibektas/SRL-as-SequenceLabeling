from numpy.core.numeric import outer
from semantictagger.conllu import CoNLL_U
from .datatypes import *
from typing import Callable , Dict, Literal , Tuple , Union

from .edgedelegate import EdgeDelegate
from .node import Node
from .edge import Edge
from .tagcandidate import TagCandidate
from .conllu import CoNLL_U

import numpy as np
import pdb 

from semantictagger import edge

from semantictagger import edgedelegate

NUMPYGT = np.greater
NUMPYLT = np.less 

SentenceDict = Dict[Index, Node]



class Span:
    def __init__(self, start : Index ,end : Index, level : int , tag : Tag):
        self.start= start
        self.end = end
        self.leveltag : List[Tuple[Tag,int]] = [(tag,level)]
        self.children : List[Span] = []
    
    def nearestgreatestspan(self , index : int , crole : Tag , level : int):
        if index == 0 :
            start = self.start
            end = self.children[0].start-1   
            if end < start:
                return False
            self.addchild(Span(start , end , level , crole))
        else:
            start = self.children[index-1].end + 1 
            end = self.children[index].start - 1
            if end < start:
                return False
            self.addchild(Span(start , end , level , crole))
        
        return True
        
        

    def addchild(self ,child):
        for i , v in enumerate(self.children):
            if child.end < v.start:
                self.children = self.children[0:i-1] + [child] + self.children[i::]
                return 
            else : 
                continue
        
        self.children.append(child)

    def add(self , span):
        if self.start < span.start:
            if self.end < span.end:
                return False
        elif self.start == span.start:
            if self.end < span.end:
                span.addchild(self)
                return span
            elif self.end == span.end:
                self.leveltag += span.leveltag
                return self
        else :
            if self.end <= span.end:
                span.addchild(self)
                return span
            else: 
                return False

        for index , child in enumerate(self.children):
            retval = child.add(span)
            if retval != False:
                if retval == span : 
                    self.children[index] = span
                return self
        
        self.addchild(span)
        return self
                




class ReconstructionModule:
    def __init__(self):

        # self.submodules = submodules
        self.roletagdict = None
        self.deptagdict = None
        self.invroletagdict = None
        self.invdeptagdict = None

        self.edgedelegate = EdgeDelegate()
        self.sentence : SentenceDict = None
        self.entry : CoNLL_U = None
        self.spandict : Dict[Index , List[Span]] = None 

    def resolvedeptag(self , index):
        return self.invdeptagdict[index]

    def resolveroletag(self, index):
        return self.invroletagdict[index]

    def resolvetag(self , tag :TagCandidate) -> Tag:
        return ("<" if tag.direction == Direction.LEFT else ">") * tag.distance + self.resolveroletag(tag.tag)

    def iscontinuationrole(self,role : Tag) -> Union[Tag,Literal[False]]:
        return role.lstrip("R-") if role.startswith("R") else False

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


    def loadsentence(self , entry : CoNLL_U):
        
        heads = entry.get_heads()
        
        self.entry = entry
        self.sentence = {i : Node(i , -1) if i == -1 else Node(i,heads[i]) for i in range(-1 ,len(entry))}
        self.edgedelegate.clear()
        self.edgedelegate.loadsentence(self.sentence)
        self.deptagdict = {}
        self.roletagdict = {}
        self.invdeptagdict = {}
        self.invroletagdict = {}
        self.spans = None 
        
        verblocs = entry.get_verb_indices()
        

        ROOT = self.sentence[-1]
        assert(ROOT.index == -1)

        if len(verblocs) == 0:
            return ["_"]*len(entry)

        for i in verblocs: 
            edge  = Edge(i , ROOT.index ,Roletag(self.maprole("V")) , Deptag(self.mapdependency("_")) , distance=-1 , direction=-1)
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

        
    def reconstruct(self):
        
        def subroutine(root : Span):
            for i in root.leveltag:
                tag = i[0]
                level = i[1]
                if tag == None :
                    break 
                start = root.start
                end = root.end
                startdelimiter = False
                for j in range(start,end+1):
                    if j == start:
                        startdelimiter = True
                        if self.spans[level][j] != "*" and not self.spans[level][j].startswith(f"({tag}*"):
                            a = 1234

                        self.spans[level][j] = f"({tag}*" if start != end else f"({tag}*)"
                        if start == end : 
                            startdelimiter = False
                        
                    elif j == end:
                        startdelimiter = False
                        self.spans[level][j] = f"*)"
                    else :
                        self.spans[level][j] = "*"
                if startdelimiter == True:
                    a = 1239
            
            for i in root.children:
                subroutine(i)
                    

        
        self.RULE3()
        self.parsespans()
        self.RULE4()

        self.spans : Annotation = self.entry.get_span()
        subroutine(self.rootspan)
        return self.spans

        # return self.entry.get_span()



    def parsespans(self):
        self.spans : Annotation = self.entry.get_span()
        verblocs = self.entry.get_verb_indices()        
        self.rootspan = Span(0,len(self.entry),-1,None) 

        for index , level in enumerate(self.spans):
            for wordindex , word in enumerate(level):
                if word.startswith("("):
                    start = wordindex
                    tag = word.strip(")(*")
                if word.endswith(")"):
                    end = wordindex
                    self.rootspan = self.rootspan.add(Span(start,end ,index , tag))
        
       
        
    
    def RULE1(self):
        """
        If there is an ARG1 and ARG2 then there is probably and ARG0. 
        TODO : How often does this assumption hold?
        """
        pass
    
    def RULE2(self):
        """
        TODO : This is also contrary to common belief not true.
        There cant be multiple role entries for main roles ARGX.
        """
        pass
    
    def RULE3(self):
        """
        If a word has a merge tag , then merge it to the closest role word of the verb it is connected to.
        """ 
        verblocs = self.entry.get_verb_indices()
        for vindex , i in enumerate(verblocs):
            verb = self.sentence[i]
            incoming = verb.incoming.copy()
            for j in incoming:
                edge : Edge = self.edgedelegate.get(j)
                if edge.roletag == self.maprole("+"):

                    incomings = [x for x in incoming if x!=j]
                    temparray = [np.abs(self.edgedelegate.get(x).from_ - edge.from_) for x in incomings]

                    if len(incomings) == 0 :
                        head = i 
                    else :
                        closest = np.argmin(temparray)
                        head = self.edgedelegate.get(incomings[closest]).from_ 


                    self.entry.updatehead(edge.from_,head)
                    self.entry.updatesrl(vindex, edge.from_,"_")
                    self.sentence[edge.from_].removeoutgoing(j)
                    self.sentence[edge.to_].removeincoming(j)
        


    def RULE4(self):
        """
            Continuation roles need role tags they follow. 
        """

        def subroutine(root):
            roles = [[tag for tag in child.leveltag] for child in root.children]

            for index , child in enumerate(root.children):
                for tagindex , tag in enumerate(child.leveltag):
                    crole = self.iscontinuationrole(tag[0])
                    if crole != False:
                        for i in range(0,index):
                            outerloopcanbreak = False
                            for leveltag in roles[i]:
                                if (crole,tag[1]) == leveltag:
                                    outerloopcanbreak = True
                                    break
                            if outerloopcanbreak : break
                        else:
                            root.nearestgreatestspan(index , crole , tag[1])
            
            for i in root.children:
                subroutine(i)

        subroutine(self.rootspan)
        









        

