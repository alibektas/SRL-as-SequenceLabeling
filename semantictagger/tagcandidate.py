import numpy as np
from .datatypes import * 



class TagCandidate:
    def __init__(self , direction , distance , tag , point : np.float):
        self.distance = distance
        self.direction = direction
        self.tag = tag 
        self.point = 0

    