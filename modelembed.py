from flair.embeddings import TokenEmbeddings
from flair.data import Sentence , Token
from typing import Union , List

import torch
import flair
from flair.models import SequenceTagger
from semantictagger.paradigms import DIRECTTAG


class ModelEmbeddings(TokenEmbeddings):

    def __init__(self):
        
        super(ModelEmbeddings, self).__init__()

        self.predictionmodel : SequenceTagger = SequenceTagger.load('./models/best-model.pt')
        self.__embedding_length = 181
        self.name = "ModelEmbeddings"

        
    @property
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return self.__embedding_length


    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        
        with torch.no_grad():
            embeds = self.predictionmodel.forward(sentences)

        for i, sentence in enumerate(sentences):
            for index , token in enumerate(sentence.tokens):
                token.set_embedding(self.name, embeds[i][index])
        
        return sentences
