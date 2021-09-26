import numpy as np 
from typing import NewType , List
import enum

Tag = str
Index = np.int
EdgeID = np.int
Annotation = List[List[Tag]]
Roletag = NewType('Roletag' , np.int)
Deptag = NewType('Deptag' , np.int)
Point = np.float

class Direction(enum.Enum):
    LEFT : np.byte = -1
    RIGHT : np.byte = 1
    ALIGNED : np.byte = 0 


class Outformat(enum.Enum):
    CONLL05 = 1
    CONLL09 = 2
    ALL = 10