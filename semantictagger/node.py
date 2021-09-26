from typing import List 

from .datatypes import * 
from .tagcandidate import TagCandidate



class Node:
    def __init__(self , index : Index , head : Index):
        self.index = index
        self.head = head
        self.tags : List[TagCandidate] = []
        self.incoming : List[EdgeID] = []
        self.outgoing : List[EdgeID] = []
        

    def addincoming(self , id : EdgeID):
        self.incoming.append(id)

    def addoutgoing(self , id : EdgeID):
        self.outgoing.append(id)

    def removeincoming(self , id : EdgeID):
        self.incoming.remove(id)

    def removeoutgoing(self , id : EdgeID):
        self.outgoing.remove(id)

    def addtag(self , tag : TagCandidate):
        self.tags.append(tag)
    
    def isheadword(self):
        return len(self.tags) > 0 


class ExtendedNode(Node):
    def __init__(self , index : Index , head : Index , postag : str):
        super().__init__(index , head)
        self.postag = postag
    
    