from .datatypes import *

class Edge:

    def __init__(
            self , 
            from_ : Index , 
            to_ : Index , 
            roletag : Roletag , 
            deptag: Deptag , 
            distance : np.int , 
            direction : Direction
        ):

        self.from_ = from_
        self.to_ = to_
        self.roletag = roletag
        self.deptag = deptag
        self.distance = distance 
        self.direction = direction
    