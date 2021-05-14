from paradigms import DIRECTTAG
from dataset import Dataset
from pathlib import Path 

fp = Path("../UP_English-EWT/en_ewt-up-train.conllu")
dataset = Dataset(fp)

# Get 10th sentence
entry = dataset[10]

# Get all SRL annotations
# Depth indicates how many verbs there are.
annotations = [entry.get_srl_annotation(d) for d in range(entry.depth)]

#Init tagger
dirtag = DIRECTTAG(2, omitlemma=True)

# Encode and decode
encoded = dirtag.encode(entry)
decoded = dirtag.decode(encoded)

#Or simply call
# Which returns how many labeled could be retrieved once
# the tagger is applied to the given entry
countcorrecttags , countfalsetags = dirtag.test(entry)

# Returns
# -------
# Sparsity :How many empty tags there are compared to nonempty ones.
# Mean of the set (emptytag excluded)
# Std (emptytag excluded)
# Frequency dictionary
# Results are printed if show_results = True
sparsity , mean , std , dict_ = dataset.getlabelfrequencies(dirtag ,show_results = True , returndict = True)

from datastats import collectaccuracy

"""
    Collects info from test results over entire dataset.

    Return
    ------
    correct : How many sentences correctly labeled
    false : How many sentences incorrectly labeled
    singlecorrect : How many tokens correctly labeled
    singlefalse : How many tokens incorrectly labeled

"""
(correct , false), (singlecorrect ,singlefalse) = collectaccuracy(dirtag , dataset , showresults= True)