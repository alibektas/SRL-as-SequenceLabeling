from typing import Dict
from .datatypes import * 
from .edge import Edge
from .node import Node

class EdgeDelegate:
    def __init__(self):
        self.dict_ = {}
        self.counter = 0
        self.sentence = None

    def loadsentence(self,sentence :  Dict[Index , Node]):
        self.sentence = sentence

    def add(self, edge : Edge):
        self.dict_[self.counter] = edge
        assert(self.sentence is not None)
        self.sentence[edge.from_].addoutgoing(self.counter)
        self.sentence[edge.to_].addincoming(self.counter)
        self.counter += 1
    
    def remove(self , id : EdgeID):
        self.dict_.pop(id)

    def clear(self):
        self.dict_ = {}
        self.counter = 0 

    def get(self , id):
        return self.dict_[id]
    