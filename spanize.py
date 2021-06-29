import re 
from flair.data import Span , Token , Sentence
from typing import List

def spanize(sentence : Sentence) :
    spans : List[Span] = []
    tokens : List[List[Token]] = []
    indices : List[int] = []

    numofmarkers : int = -1
    direction : int = -1
    nearesthead : int = -1 

    regex = re.compile("<+|>+")

    for i in range(len(sentence.tokens)):
        tokencur = sentence.tokens[i]
        role = tokencur.get_labels("role")[0].value
        verb = tokencur.get_labels("verb")[0].value


        if (role != "" and not regex.fullmatch(role)) or verb=="V":
            indices.append(i)
            tokens.append([tokencur])



    for i in range(len(sentence.tokens)):
        tokencur = sentence.tokens[i]
        verb = tokencur.get_labels("verb")[0].value
        role = tokencur.get_labels("role")[0].value

        if regex.fullmatch(role):
            numofmarkers = len(role)-1
            direction = -1 if role[0] == "<" else 1

            if i <= indices[0]:
                nearesthead = 0
                index = -numofmarkers if direction == -1 else numofmarkers+1
            else:
                for j in range(len(indices)) :
                    if j < len(indices) - 1:
                        if indices[j] <= i and i <= indices[j+1]:
                            nearesthead = j
                            break
                    else :
                        if indices[j] <= i:
                            nearesthead = j
                            break

                index = j+(-numofmarkers if direction == -1 else numofmarkers+1)

            tokens[index].append(tokencur)

        elif role == "":
            continue
        else :
            continue
    
    return tokens