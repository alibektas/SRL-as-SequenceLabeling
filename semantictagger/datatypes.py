import numpy as np 
from typing import NewType
import enum

Tag = str
Index = np.int
EdgeID = np.int
Roletag = NewType('Roletag' , np.int)
Deptag = NewType('Deptag' , np.int)
Point = np.float


class Direction(enum.Enum):
    LEFT : np.byte = 1
    RIGHT : np.byte = -1
    ALIGNED : np.byte = 0 
