from flair.embeddings import Embeddings
from flair.data import Sentence
from typing import Union , List

import torch
import flair
from flair.models import SequenceTagger
from semantictagger.paradigms import DIRECTTAG


class VerbEmbedding(Embeddings):

    verbprediction : SequenceTagger = SequenceTagger.load('./modelout/verbonly/best-model.pt')


    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        vencoded = self.verbprediction.predict(sentences)
        
        for i, sentence in enumerate(sentences):

            for token, token_idx in enumerate(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = torch.FloatTensor([1], device= flair.device) if vencoded[index] == "V" else torch.FloatTensor([1], device= flair.device)

                token.set_embedding(self.name, word_embedding)

        return sentences


        for sentence in sentences:
            verbenc = self.dirtag.encode(sentence)
            for index in range(len(sentence)):
                if verbenc[index] == "V":
                    sentence[index]._embeddings[self.name] = 

