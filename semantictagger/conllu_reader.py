"""
    Reads a CoNLL-U file. Elliptic analysis is not a part of the parsing scheme.
    TODO : The sentence "unique gifts and cards" is a problem
"""
from . import conllu
import uuid 

class CoNLL_Reader():
    def __init__(self , path):
        self.entries =[]
        self.file_pointer = open(path)
        for entry in self.read_sentence():
            self.entries.append(entry)
        self.file_pointer.close()
        
        
    def read_sentence(self):
        entry = []
        while True:
            line = self.file_pointer.readline()
             
            if len(line) == 0:
                return None
            else:
                if(line.startswith("#")):                  
                    continue
                elif line == "\n":
                    yield conllu.CoNLL_U(entry , uuid.uuid4())
                    entry = []

                else :
                    elems = line.replace("\n" , "").split("\t")
                        
                    if "." in elems[0]:
                        # See analysis of ellipsis at https://universaldependencies.org/format.html
                        continue
                
                    if elems[11] == "":
                        # Fill at least with underscore if the sentence is not annotated.
                        elems[11] = "_"
                    
                    if elems[10] == "":
                        elems[10] = "_"
                    
                    if elems[10] != "_":
                        temp = elems[10].split(".")
                        
                        if not temp[1].startswith("0"):
                            # TODO : Remove this in the future.
                            temp[1] = "0" + temp[1]
                            elems[10] = ".".join(temp)
                        
                        
                    dict_ = {
                        "form" : elems[1] ,
                        "lemma" : elems[2] ,
                        "upos" : elems[3] ,
                        "xpos" : elems[4],
                        "feats" : elems[5],
                        "head" : elems[6],
                        "deprel" : elems[7],
                        "vsa":elems[10],
                        "srl" : elems[11:]
                    }     
                    entry.append(dict_)
                