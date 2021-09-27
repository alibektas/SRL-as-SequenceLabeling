from semantictagger.dataset import Dataset
from semantictagger.paradigms import FRAMETYPE , POSTYPE , Encoder
import time
import os 
from tqdm import tqdm
import pdb

def writecolumncorpus(
        dataset : Dataset , 
        encoding : Encoder = None,
        frame_gold = False,
        pos_gold = False,
        filename = None,
        frameonly = False,
        posonly = False,
        postype : POSTYPE= POSTYPE.UPOS,
        downsample = False 
    ):
    
    if filename is not None:
        filename_ = filename   
    else:
        filename_= time.time_ns()
    
    progress = 0 
    total = len(dataset.entries)
    if downsample != False:
        total = downsample
    dirname = os.path.dirname(__file__)
    if frameonly : frame_gold = True
    if posonly : pos_gold = True
    if frameonly == posonly == True: 
        print("Cant create columncorpus files when posonly and frameonly are set to True at the same time.")
        raise AssertionError()



    with open(f"{dirname}/data/{filename_}.tsv" , 'x') as fp :
        for index in tqdm(range(total)):
            sentence = dataset.entries[index]
            
            if encoding is not None:
                encoded = encoding.encode(sentence)
            
            if frame_gold:
                frames = sentence.get_vsa()
                if encoding.frametype == FRAMETYPE.PREDONLY:
                    frames = ["V" if i != "_" else "" for i in frames]
                elif encoding.frametype == FRAMETYPE.FRAMENUMBER:
                    frames = [f"V.{i[-2:]}" if i != "_" else "" for i in frames]

            if pos_gold:
                if encoding.postype == POSTYPE.UPOS:
                    pos = sentence.get_by_tag("upos")
                else:
                    pos = sentence.get_by_tag("xpos")


            
            words = sentence.get_words()
            assert(len(encoded)== len(words))
            for i in range(len(words)):
                if frameonly:
                    fp.write(f"{words[i]}\t{frames[i]}\n")
                elif posonly:
                    fp.write(f"{words[i]}\t{pos[i]}\n")
                elif encoding is not None:    
                    fp.write(f"{words[i]}\t{encoded[i]}")
                    if frame_gold:
                        fp.write(f"\t{frames[i]}")
                    if pos_gold:
                        fp.write(f"\t{pos[i]}")
                    fp.write("\n")
            fp.write("\n")
    
    print(f"SUCCESS : File /data/{filename_}.txt written.")
