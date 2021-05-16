"""
    Container for a single CoNLL_U entry.
"""
from spacy import displacy
from pathlib import Path
import pdb 
import uuid
import os

USING_JUPYTER = False
class CoNLL_U():
    def __init__(self , content , id):
        self.content = content
        self.depth = len(content[0]['srl'])
        self.__len = len(self.get_words())
        self.id = id

    def __len__(self):
        return self.__len

    def get_vsa(self):
        """ Returns Verb Sense Annotation."""
        return [part['vsa'] for part in self.content]
            
    def get_srl_annotation(self , depth):
        if depth < 0 or depth > self.depth:
            IndexError()
        
        return [part['srl'][depth] for part in self.content]

       
    def info(self , srl = True):
        print("SENTENCE:")
        print("----------------------------")
        print(self.get_sentence())
        print("\n")

    
        print("DEPTH:")
        print("----------------------------")
        print(self.depth)
        print("\n")

        if srl:
            print("SRL ANNOTATIONS")
            print("----------------------------")
            for i in range(self.depth):
                print(f"Depth ({i}) : {self.get_srl_annotation(i)}")


    def get_sentence(self):
        return " ".join(self.get_words())

    def get_words(self):
        return [part['form'] for part in self.content]
    
    def get_by_tag(self, tag):
        """ Get one column of the UD entries."""
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



    def get_local_tags(self , verb_distance , add_verb_lemma = False,  abs = False):
        
        NotImplementedError()
        
        annotations = [self.get_srl_annotation(d) for d in range(self.depth)]
        tags = [""] * len(self.get_words())
        vsa = self.get_vsa()


        root = None
        current = None
        verb_locs = []
        
        for row in range(len(annotations)):
            index_of_verb = -1
            encodings = []
            for col in range(len(annotations[row])):
                if annotations[row][col] == "V":
                    index_of_verb = col
                    if add_verb_lemma:
                        tags[col] = vsa[col]
                    else :
                        tags[col] = vsa[col][-2::]
                elif  annotations[row][col] == "_":
                    continue
                else :
                    encodings.append((col,annotations[row][col]))
            
            if root is None :
                root = VC(index_of_verb, vsa[index_of_verb] , verb_distance)
                current = root
            else :
                current.next = VC(index_of_verb, vsa[index_of_verb] , verb_distance)
                current = current.next
            
            current.add(0,encodings)
            encodings = []
        
        root.stabilize()
        
    def get_verb_indices(self):
        return [index for index , value in enumerate(self.get_vsa()) if value != '_' and value != '_=' ]
    
    
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



            
class VC():
    def __init__(self , vindex, vsa , distance , next = None):
        self.vsa = vsa
        self.index = vindex
        self.locs = [[]] * distance
        self.next = next
    
    def __str__(self):
        pass

    def add(self , level , content):
        self.locs[level].append(content)

    def stabilize(self):
        for index , level in enumerate(self.locs):
            for elem in level:
                if elem[0] in [x[0] for x in self.next.locs[0]]:
                    if index+1 < len(self.locs):
                        self.next.locs[index+1].append(elem)
                        level.remove(elem)
        
        if self.next:
            self.next.stabilize()
 

