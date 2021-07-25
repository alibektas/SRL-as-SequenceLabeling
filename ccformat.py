from semantictagger.dataset import Dataset
from semantictagger.paradigms import Encoder
import time
import os 
from tqdm import tqdm

def writecolumncorpus(dataset : Dataset , encoding : Encoder , filename = None , verbmarker = False , verbsonlyencoder= None):
    
    if filename is not None:
        filename_ = filename   
    else:
        filename_= time.time_ns()
    
    progress = 0 

    total = len(dataset.entries)
    dirname = os.path.dirname(__file__)

    with open(f"{dirname}/data/{filename_}.txt" , 'x') as fp :
        for index in tqdm(range(total)):
            sentence = dataset.entries[index]
            encoded = encoding.encode(sentence)
            
            if verbsonlyencoder is not None:
                verbsencoded = verbsonlyencoder.encode(sentence)
            
            words = sentence.get_words()
            assert(len(encoded)== len(words))
            for i in range(len(words)):
                if verbmarker and verbsencoded[i] == "V" :
                    fp.write(f"{words[i]}*\t{encoded[i]}\n")
                else :
                    fp.write(f"{words[i]}\t{encoded[i]}\n")
            fp.write("\n")
    
    print(f"SUCCESS : File /data/{filename_}.txt written.")
