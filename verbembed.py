from flair.embeddings import TokenEmbeddings
from flair.data import Sentence
from typing import Union , List

import torch
import flair
from flair.models import SequenceTagger
from semantictagger.paradigms import DIRECTTAG

import pdb 


class VerbEmbedding(TokenEmbeddings):

    verbprediction : SequenceTagger = SequenceTagger.load('./modelout/verbonly/best-model.pt')


    def __init__(self):
        self.__embedding_length = 1
        self.name = "VerbPrediction"
        super().__init__()

    @property
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return self.__embedding_length


    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        
        self.verbprediction.predict(sentences , label_name="VSA") # Pretrained model for Verb Sense Annotation

        for i, sentence in enumerate(sentences):
            for index , token in enumerate(sentence.tokens):
                isverb = token.get_labels("VSA")[0].value == "V" # See if model tagged token with 'V' tag. 
                word_embedding = torch.FloatTensor([1]) if isverb else torch.FloatTensor([0])
                token.set_embedding(self.name, word_embedding)
                pdb.set_trace()

        return sentences