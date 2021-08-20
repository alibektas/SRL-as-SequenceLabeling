from semantictagger.datatypes import Tag
from .tagcandidate import TagCandidate
from .datatypes import *
from typing import List


class SelectionDelegate:
    def __init__(self , rules):
        self.rules = rules

    def select(self , candidates : List[TagCandidate]) -> TagCandidate:
        
        cand = candidates
        for i in self.rules:
            cand = i(cand)

        assert len(cand) == 1
        return cand[0]

