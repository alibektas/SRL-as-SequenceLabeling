from typing import Dict, Mapping , Union , List

from torch.utils.data.dataset import Dataset
from . import paradigms
from . import dataset

class MapProtocol():
    def __init__(self ,lowkey : int = -1 , unknw : 'str' = "<UNKNW>"):
        self.lowkey = lowkey
        self.unkwn = unknw

class Mapper():
    def __init__(self ,datasets : Union[Dataset, List[dataset.Dataset]] , 
        encoder : paradgims.Encoder, 
        protocol : MapProtocol,
        percentage : int =  5):
        
        if type(datasets) == Dataset:
            datasets = [datasets]
        
        self.datasets = datasets
        self.encoder = encoder
        self.protocol = protocol
        self.percentage = percentage
        self.mapping = {}
        self.initmaptag()

    def initmaptag(self): 

        for dataset in self.datasets:
            for sentence in dataset:
                encoded = self.encoder.encode(sentence)
                for i in encoded : 
                    if not i in self.mapping :
                        self.mapping[i] = 1 
                    else :
                        self.mapping[i] += 1

        
        medval =  sum(self.mapping) // len(self.mapping)
        minval = min(self.mapping.keys())

        fiftyper = medval - minval
        lowerbound = fiftyper // (50 // self.percentage)

        for i in self.mapping.keys():
            if self.mapping[i] <= lowerbound :
                self.mapping[i] = self.protocol.lowkey
            else : 
                continue
    
    def map(self,tag : str):
        return self.mapping[tag] if self.mapping[tag] != self.protocol.lowkey else self.protocol.unknw