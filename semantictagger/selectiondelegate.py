from semantictagger.datatypes import Tag
from .tagcandidate import TagCandidate
from .datatypes import *
from typing import Callable, List


class SelectionDelegate:
    def __init__(
        self , 
        rules : List[Callable[[List[TagCandidate]] , List[TagCandidate]]]
        ):
        self.rules = rules

    def select(self , candidates : List[TagCandidate]) -> TagCandidate:
        
        cand = candidates
        for i in self.rules:
            cand = i(cand)
            if len(cand) == 1:
                # TODO 
                """
                    It surely makes sense to terminate with this condition. Yet it may be also be 
                    the case that we want to see it to the end , as residual rules may be of "confirming" nature.
                """
                break

        assert len(cand) == 1
        return cand[0]


