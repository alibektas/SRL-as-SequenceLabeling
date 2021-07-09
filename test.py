from pathlib import Path
from semantictagger.dataset import Dataset
from semantictagger import paradigms
from flair.data import Sentence

dev_file = Path("./UP_English-EWT/en_ewt-up-dev.conllu")
dataset_dev = Dataset(dev_file)
relpos = paradigms.RELPOS()
dirtag = paradigms.DIRECTTAG(
            mult = 2, 
            rolehandler = 'complete',
            verbshandler  = 'omitverb',
            verbsonly = False,
            deprel = True,
            depreldepth = 5)

entry = dataset_dev[2]
words = entry.get_words()

encoded = dirtag.encode(entry)
for i , v in enumerate(encoded):
    print(words[i] , v) 