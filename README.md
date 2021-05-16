# README

Dataset is read by ``CoNLL_Reader``. You can directly reference a dataset entry by subscribing with entry index. To use an encoding scheme you need to initialize one under``paradigms.py``. The most recent scheme that is of researcher's interest is ``paradigms.DIRECTTAG``. A word's role to a predicate is indicated by labeling the role explicitly and stating as a prefix of ``(>){1,mul}|(<){1,mul}``  This takes two parameters in initialization. First one ``mul`` indicating the furthest distance a verb can be away from a given label in terms of verb distance (how many other verbs are in between). Second one is ``ommitlemma`` depending on which verb's lemma is not included but only its sense (e.g ``V01`` instead of ``Vmake.01``). 

```python
    from semantictagger.paradigms import DIRECTTAG
    from semantictagger.dataset import Dataset
    from pathlib import Path 

    fp = Path("./UP_English-EWT/en_ewt-up-train.conllu")
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

    from semantictagger.datastats import collectaccuracy

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
```

### TODO
1. Dataset entry no 269 has two verbs in one annotation level. How to solve this problem?
2. Sentence no. 5081 has two exactly same semantic role layers (verbs coincide). What to do with it ?
