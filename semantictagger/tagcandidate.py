import numpy as np
from numpy.core.defchararray import index
from .datatypes import * 



class TagCandidate:
    def __init__(self , direction , distance , tag , point : np.float , indexframe : np.int):
        self.distance = distance
        self.direction = direction
        self.tag = tag 
        self.point = 0
        self.indexframe = indexframe

    