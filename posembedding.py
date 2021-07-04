from flair.embeddings import TokenEmbeddings
from flair.data import Sentence , Token
from typing import Union , List

import torch
import flair
from flair.models import SequenceTagger
from semantictagger.paradigms import DIRECTTAG


class POSEmbeddings(TokenEmbeddings):

    def __init__(self):
        
        super(POSEmbeddings, self).__init__()
        
        self.news_f = "0-/home/alan/.flair/embeddings/news-forward-0.4.1.pt"
        self.news_b = "1-/home/alan/.flair/embeddings/news-backward-0.4.1.pt"


        self.tagger = SequenceTagger.load("flair/pos-english")
        self.__embedding_length = 53
        self.name = "POSEmbedding"

    @property
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return self.__embedding_length


    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        
        with torch.no_grad():
            embeds = self.tagger.forward(sentences)

        for i, sentence in enumerate(sentences):
            for index , token in enumerate(sentence.tokens):
                token.set_embedding(self.name, embeds[i][index])
                token.clear_embeddings(embedding_names = [self.news_f , self.news_b])
    
        return sentences
