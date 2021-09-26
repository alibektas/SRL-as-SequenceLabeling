import dataset
from typing import List , Union , Tuple


class MapProtocol():
    def __init__(self ,lowkey : int = -1 , unkwn : 'str' = "<UNKNW>"):
        self.lowkey = lowkey
        self.unkwn = unkwn

class Mapper():
    def __init__(self ,datasets : Union[dataset.Dataset, List[dataset.Dataset]] , 
        encoder , 
        protocol : MapProtocol,
        percentage : int =  5 , 
        lowerbound : int = -1):
        
        if type(datasets) == dataset.Dataset:
            datasets = [datasets]
        
        self.datasets = datasets
        self.encoder = encoder
        self.protocol = protocol
        self.percentage = percentage
        self.lowerbound = lowerbound
        self.usebound = True if self.lowerbound != -1 else False
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

        if not self.usebound:
            medval =  sum(self.mapping.values()) // len(self.mapping)
            minval = min(self.mapping.values())

            fiftyper = medval - minval
            self.lowerbound = fiftyper // (50 // self.percentage)
        
        print(f"Discarding all elements which have less than {self.lowerbound} occurences.")

        for i in self.mapping.keys():
            if self.mapping[i] <= self.lowerbound :
                self.mapping[i] = self.protocol.lowkey
            else : 
                continue
    
    def maptag(self,tag : str):
        if self.mapping[tag] != self.protocol.lowkey:
            return tag

        return tag.split("|")[2]


class PairMapper:
    """
        A simple class to keep things ordered.
        Some roles occur at the same time quite often.
        To see the exact numbers please refer to 
        dataset.Dataset.findcommonroles 

    """
    def __init__(self , roles : List[Tuple[str,str,str]]):
        self.usedcount = 0 
        self.roles = roles

    def map(self , ann : List[str]) -> Union[str,bool]:
        for i in range(len(ann)):
            for j in range(i+1,len(ann)):
                for role in self.roles:
                    if role[0] == ann[i] and role[1]==ann[j]:
                        self.usedcount += 1
                        return role[2]
        
        return False

    def __str__(self) -> str:

        _str = "PAIRMAP \n"
        for i in self.roles:
            _str += f"\t {i[0]} , {i[1]} -> {i[2]} \n"
        _str +=  f"PairMapper paired {self.usedcount} times."

        return _str