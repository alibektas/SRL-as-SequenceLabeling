"""
    Container for a single CoNLL_U entry.
"""
from semantictagger.datatypes import Annotation
from spacy import displacy
from pathlib import Path
import os


import numpy as np 


BEGIN = 1
INSIDE = 2
OUT = 3

USING_JUPYTER = False
class CoNLL_U():
    def __init__(self , content):
        self.content = content
        self.depth = len(content[0]['srl'])
        self.__len = len(self.get_words())

        self.vlocs = np.array([index for index , value in enumerate(self.get_vsa()) if value != '_' ] , dtype=np.int)
        self.headlocs = np.array([int(x)-1 for x in self.get_by_tag("head")])

    def updatehead(self, index , newhead):
        self.headlocs[index]=newhead
        self.content[index]['head'] = str(newhead+1)

    
    def updatesrl(self , level , index , newlabel):
        self.content[index]['srl'][level] = newlabel

    def __len__(self):
        return self.__len


    def get_vsa(self):
        """ Returns Verb Sense Annotation."""
        return [part['vsa'] for part in self.content]
            

    def get_pos(self):
        return self.get_by_tag("xpos")

    def is_predicate(self , index):
        """
            Does the the element with the questioned index act as a predicate for any annotation level?
            ::return:: 
            True , if so.
            False , otherwise. 
        """
        return index in self.vlocs

    def get_srl_annotation(self , depth = None) -> Annotation:
        """
        Gets n-th level of the SRL annotations. 
        If depth is not specified , then all annotational levels will be returned.
        Index Error if the passed depth doesn't exist. 
        """
        if depth is None:
            return [self.get_srl_annotation(d) for d in range(self.depth)]

        if depth < 0 or depth > self.depth:
            IndexError()
        
        return [part['srl'][depth] for part in self.content]


    def get_depbased(self):
        srl = self.get_srl_annotation()

        for i , v in enumerate(srl):
            for j , v2 in enumerate(v):
                if v2 == "_":
                    srl[i][j] = "*"
                else :
                    srl[i][j] = "(" + srl[i][j] + "*)"

        return srl 


    def get_span(self) -> Annotation:
        """ Get CoNLL-05 like span-based annotations for each predicate level."""
        vlocs = [i for i  , v in enumerate(self.get_vsa()) if v != "_" and v != ""]
        head = 0 
        spans : Annotation = [["*" for i in range(len(self))] for j in range(len(vlocs))]
        heads = self.get_heads()      

 
        for i1 , verbindex in enumerate(vlocs):
            srl = self.get_srl_annotation(i1)
            arglocs = [i for i , v in enumerate(srl) if v != "_"]
            references = [-1 for _ in range(len(self.content))]

            for i2 , words in enumerate(self.content):
                
                node = i2
                head = heads[node]
                loop_detection = len(self.content)

                while True:
                    if node in arglocs:
                        if node == verbindex :
                            if i2 == verbindex :
                                references[i2] = node
                            else :
                                references[i2] = -1
                        else :
                            references[i2] = node
                        break
                    else :
                        node = head
                        loop_detection -= 1
                        if node == -1 : break
                        
                        if node < len(heads):
                            head = heads[node]
                        else :
                            head = -1

                        if loop_detection == 0  :
                            references[i2] = i2 
                            break


            refindex = 0 # where the dependency head lies
            curindex = 0 
            label = ""
            bio = BEGIN
            counter = 0 
            endcondition = len(self)
            startindex = 0 
            while curindex < endcondition:
                if references[curindex] != -1 :
                    if bio == BEGIN:
                        refindex = references[curindex]
                        if srl[refindex] == "_":
                            curindex += 1
                            continue
                        startindex = curindex
                        if refindex < curindex:
                            curindex += 1
                            continue
                        bio = INSIDE

                    elif bio == INSIDE :
                        if refindex != references[curindex]:
                            if refindex <= curindex:
                                spans[i1][startindex] = f"({srl[refindex]}*"
                                spans[i1][curindex-1] += ")"
                            bio = BEGIN
                            continue
                else :
                    if bio == INSIDE :
                        if refindex != references[curindex] and refindex <= curindex:
                            spans[i1][startindex] = f"({srl[refindex]}*"
                            spans[i1][curindex-1] += ")" 
                        bio = BEGIN 
            
                curindex += 1
            
            if bio == INSIDE:
                spans[i1][startindex] = f"({srl[refindex]}*"
                spans[i1][endcondition-1] += ")"
        

        return spans

                

    def get_sentence(self):
        return " ".join(self.get_words())

    def get_words(self):
        return [part['form'] for part in self.content]
    
    def get_by_tag(self, tag):
        """
        Get one column of the UD entries.
        Possible tags are : ["form","lemma","upos","xpos","feats","head","deprel","vsa","srl"]
        """
        return [part[tag] for part in self.content]
    
    def get_role_density(self):
        annotation_density = [0]*len(self.content) 
        for i in range(self.depth):
            srl = self.get_srl_annotation(i)
            for index , value in enumerate(srl):
                if(value != '_'):
                    annotation_density[index] += 1
        
        
    def visualize_dep(self , writeToFile = False , USING_JUPYTER= True):
        tree = { "words": [{"text" : elem['form'] , "tag" : elem['xpos']} for elem in self.content],
            "arcs": [{
                "label":elem['deprel'] ,
                "start":index , 
                "end": int(elem['head'])-1 if int(elem['head']) != 0 else index ,
                "dir": "right" if index+1 < int(elem['head']) and int(elem['head']) != 0 else "left"} 
                for index , elem in enumerate(self.content)
            ]}
        

        
        if USING_JUPYTER:
            a = displacy.render(tree,manual=True, style='dep' , options={"distance":140})
        else :
            a = displacy.serve(tree,manual=True, style='dep' , options={"distance":140})
        
    
        if writeToFile:
            a = displacy.render(tree , jupyter=False ,manual=True, style='dep' , options={"distance":140})
            save_to_location("dep.svg" , a)

    
            
    def visualize_vsa(self , writeToFile= False):
        annot = self.get_vsa()
        entities = []
        char_level_counter = 0

        for index , word in enumerate(self.get_words()):
            if annot[index] != '_':
                entities.append({
                    "start" : char_level_counter,
                    "end" : char_level_counter + len(word) ,
                    "label" : annot[index]
                })

            char_level_counter += len(word) + 1

        tree = {
            'text': self.get_sentence() ,
            'ents': entities,
            'title': None
        }
        
        if USING_JUPYTER:
            a = displacy.render(tree , manual=True,style='ent')
        else :
            a= displacy.serve(tree , manual=True,style='ent')

    
    def visualize_srl(self , writeToFile= False):
        
        
        for depth in range(self.depth):
            annot = self.get_srl_annotation(depth)
            entities = []
            char_level_counter = 0

            for index , word in enumerate(self.get_words()):
                if annot[index] != '_':
                    entities.append({
                        "start" : char_level_counter,
                        "end" : char_level_counter + len(word) ,
                        "label" : annot[index]
                    })

                char_level_counter += len(word) + 1

            tree = {
                'text': self.get_sentence() ,
                'ents': entities,
                'title': None
            }
            
            if USING_JUPYTER:
                a = displacy.render(tree , manual=True,style='ent')
            else :
                a= displacy.serve(tree , manual=True,style='ent')


    def get_verb_indices(self):
        return self.vlocs
    
    def get_heads(self):
        return self.headlocs
    
    
    def visualize(self , show_verb_sense_annotation = True , writeToFile=None):
        print(self.get_sentence())
        self.visualize_dep(writeToFile)
        
        if show_verb_sense_annotation:
            self.visualize_vsa()
        
        self.visualize_srl()

    
    
save_location = Path("./images/")
def save_to_location(name : str , content):
    with open(os.path.join(save_location , name) , 'w+') as fp:
        fp.write(content)

            

