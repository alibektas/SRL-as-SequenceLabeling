from pathlib import Path
import pathlib
import numpy as np 
import flair
from pathlib import Path
from typing import dict , Tuple

import matplotlib.pyplot as plt

def confusion(
    path_to_tsv_file : Path , 
    tag_dictionary : dict[str,int] , 
    include_directions = True) -> (Tuple(dict[str,int] , np.ndarray)) :

    """
        :param tag_dictionary: Should be obtained by flair's corpus.make_tag_dictionary() function.
        :param include_directions: If True, two tags such as '>ARG1' and ARG1 are considered as equals.
        :return: A tuple consisting of a mapping of tags to indices and confusion matrix.
    """

    tagtoindex : dict[str , int] = {}

    if include_directions:
        tagtoindex = {x : index for index , x in enumerate(tag_dictionary)}
        d = len(tagtoindex)
        array = np.ndarray(shape=(d,d) , dtype= np.int)
    else: 
        counter = 0 

        for i in tag_dictionary:
            trimmed = i.replace("<" , "")
            trimmed = trimmed.replace(">" ,"")
            if tagtoindex[trimmed] : continue
            else:
                tagtoindex[trimmed] = counter
                counter += 1
        
        array = np.ndarray(shape=(counter,counter) , dtype=np.int)
 

    real = None # What the label should be inferred as
    infer = None # What it is inferred.
    
    with path_to_tsv_file.open() as file:
        while True:
            line = file.readline()
            if not line:
                return
            else :
                elems = line.split("\t")
                if len(elems) <= 2 : 
                    continue
                else:
                    real = elems[1]
                    infer = elems[2]

                    if real != infer:
                        row = tagtoindex[real]
                        col = tagtoindex[infer]
                        array[row][col] += 1

    return tagtoindex , array


def showconfusionmatrix(tagtoindex : dict[str,int] , confusionmatrix  : np.ndarray):
    ## TODO : Heat plotting.    
    pass

            
