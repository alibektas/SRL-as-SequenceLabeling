from flair.embeddings import TokenEmbeddings
from flair.data import Sentence , Token
from typing import Union , List

import torch
from flair.models import SequenceTagger
from semantictagger.paradigms import DIRECTTAG


class VerbEmbeddings(TokenEmbeddings):

    def __init__(self):
        
        super(VerbEmbeddings, self).__init__()

        self.verbprediction : SequenceTagger = SequenceTagger.load('./modelout/verbonly/best-model.pt')
        self.__embedding_length = 1 
        self.name = "VerbEmbeddings"

        
    @property
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return self.__embedding_length


    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        
        with torch.no_grad():
            embeds = self.verbprediction.forward(sentences)

        for i, sentence in enumerate(sentences):
            for index , token in enumerate(sentence.tokens):
                token.set_embedding(self.name, embeds[i][index])
        
        return sentences
