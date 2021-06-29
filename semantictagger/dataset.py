"""
    Container for a dataset.
"""
from . import conllu_reader , datastats , tag 

import matplotlib.pyplot as plt 
import numpy as np 
from pandas import DataFrame


class Dataset():
    def __init__(self , path_to_conllu):
        """ Where all entries reside"""
        self.entries = conllu_reader.CoNLL_Reader(path_to_conllu).entries
        self.size = len(self.entries)
        self.tags = {}
        self.get_unique_tags()

    def __getitem__(self,index):
        return self.entries[index]

    def __iter__(self):
        for i in self.entries:
            yield i

    def by_index(self,index):
        return self.entries[index]
    
    def visualize(self,index):
        self.by_index(index).visualize()

    def find_neighbor_commonalities(self , distance , writeFile = False):
        array = np.zeros((1,distance),dtype=np.int)
        i = 40 

        for entry in self.entries:
            annotated_indices_in_each_depth = []
            for level in range(entry.depth):
                anno = entry.get_srl_annotation(level)
                annotated_indices_in_each_depth.append([index for index , elem in enumerate(anno) if elem !='_'])

                
            for x in range(entry.depth):
                for y in range(x , min(entry.depth,distance)):
                    array[0][y-x] += len(set(annotated_indices_in_each_depth[x]).intersection(set(annotated_indices_in_each_depth[y])))

        fig, ax = plt.subplots()
        
        print(array[0])
        print(array[0][1::])
        
        ax.bar(range(1,len(array[0])),height= (array[0][1::]))


        if writeFile:
            plt.savefig("./images/locality.png")
            print("Written to file ./images/locality.png")
        
        plt.show()
            
        
    def get_depth_histogram(self , subplot):
        a = [0]*40
        for entry in self.entries:
            a[entry.depth] += 1

        subplot.bar(range(len(a)),height=a)
    
    def get_by_query(self, query):
        return [e for e in self.entries if query(e)]
    
    def get_unique_tags(self):        
        for entry in self.entries:
            for depth in range(entry.depth):
                annotation = entry.get_srl_annotation(depth)
                for elem in annotation:
                    if(elem != '_'):
                        if elem in self.tags:
                            self.tags[elem] += 1
                        else:
                            self.tags[elem] = 1
    
    def get_token_sparsity(self):
        empty_tokens = 0
        number_of_tokens = 0

        for sentence in self.entries:
            for entry in sentence.content:
                number_of_tokens += 1 
                if(all('_' == srl for srl in entry['srl'])):
                    empty_tokens +=1


        print(f'{empty_tokens/number_of_tokens} of all {number_of_tokens} tokens don\'t have a token at all.')

        
    def get_tion_suffix_info(self):
        counter = 0
        no_counter = 0

        for sentence in self.entries:
            for entry in sentence.content:
                if(entry['form'].endswith("tion")):
                    if('V' in entry['srl']):
                        counter += 1
                    else :
                        no_counter += 1


        print(f'Out of {counter+no_counter} words that end with suffix -tion %{counter/(counter+no_counter)*100} appear as verb')

    def get_absolute_density(self):
        NotImplementedError()
    
    def get_relative_density(self):
        NotImplementedError()
        
    
    def measure_information_loss(self):
        NotImplementedError()

    def filledtagfollowedbyemptytag(self , paradigm, showresults = True):
        '''
            How many of the non empty labels encoded with paradigm is followed by an empty tag?
        '''
        empty = 0 
        nonempty = 0 
        checkfornext = False

        for entry in self.entries:
            encoded = paradigm.encode(entry )
            for label in encoded :
                if label == "":
                    if checkfornext:
                        empty += 1
                elif not label.startswith("V"):
                    if checkfornext :
                        nonempty += 1
                else: 
                    if checkfornext:
                        empty += 1
                    else :
                        checkfornext = True

        if showresults:
            print(f"{empty/(nonempty+empty):.2f} is the probability that a nonempty labeled is followed by an empty label.")

        return empty/(nonempty+empty)      
                    

    def getlabelfrequencies(self , paradigm   , show_results = True , returndict = False):
        """
            Summary
            -------
            Measures label frequencies for a given encoding paradigm.
            Prints results if show_results is True.
            Results are 
            1. Sparsity (#emptylabels/#all)
            2. Mean
            3. Standard Deviation

            Results exclude empty labels' occurences.

            Returns
            -------
            Results are returned as a 3-tuple.
            If returndict is True, collected label dictionary is also returned. 
        """

        dict_ = {}
        emptytagcount = 0
        residualtagcount = 0
        
        for entry in self.entries:
            encoded = paradigm.encode(entry)
            for i in encoded:
                if i == "" or i == "_":
                    emptytagcount += 1
                    continue
                else :
                    residualtagcount += 1

                if i in dict_:
                    dict_[i] += 1
                else:
                    dict_[i] = 1
                
                
 
        sparsity = emptytagcount / (emptytagcount+residualtagcount)
        values = list(dict_.values())
        mean = np.mean(values)
        std = np.std(values)

        if show_results:
            print("\n Frequency Results For Individual Tags:")
            print("--------------------------------------")
            print(f"{sparsity*100:.2f}% of all tags are empty tags.")
            print(f"MEAN : {mean:.3f}")
            print(f"Standard Deviation  : {std:.3f}")
            
        if returndict:
            return sparsity , mean , std , dict_
        
        return sparsity , mean , std