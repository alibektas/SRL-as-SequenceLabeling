from semantictagger.dataset import Dataset
from semantictagger.paradigms import Encoder
import time
import os 

def writecolumncorpus(dataset : Dataset , encoding : Encoder):
    filename = time.time_ns()
    progress = 0 
    total = len(dataset.entries)
    dirname = os.path.dirname(__file__)

    with open(f"{dirname}/tmp/{filename}.txt" , 'x') as fp :
        for index , sentence in enumerate(dataset.entries):
            if progress <= (index / total) * 100 + 5 :
                progress += 5
                print(f"{(progress//5)*'#'}{(20-progress//5)*' '}|")

            encoded = encoding.encode(sentence)
            words = sentence.get_words()
            assert(len(encoded)== len(words))
            for i in range(len(words)):
                fp.write(f"{words[i]}  {encoded[i]}\n")
            fp.write("\n")
    
    print(f"SUCCESS : File /out/{filename}.txt written.")