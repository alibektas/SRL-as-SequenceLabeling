

def collect_stats(dict_):
     
    a = [0]*40

    for i in dict_.values():
        if i < 40:
            a[i] += 1
    
    
    return a

def collectaccuracy(encoding , dataset , showresults = True):
    """
        Collects info from test results over entire dataset.

        Return
        ------
        correct : How many sentences correctly labeled
        false : How many sentences incorrectly labeled
        singlecorrect : How many tokens correctly labeled
        singlefalse : How many tokens incorrectly labeled

    """

    correct = 0 
    false = 0 
    singlecorrect = 0
    singlefalse = 0

    for index , entry in enumerate(dataset.entries):
        correcttagged , falsetagged = encoding.test(entry)
        singlecorrect += correcttagged
        singlefalse +=  falsetagged
        if falsetagged == 0:
            correct += 1
        else :
            false += 1
    
    if showresults:
        print("\n Accuracy Results Over Entire Dataset")
        print("------------------------------------")

        print(f"{correct/(correct+false)*100:.2f}% sentences correctly labeled")
        print(f"{false/(correct+false)*100:.2f}% sentences incorrectly labeled")
        print(f"{singlecorrect/(singlecorrect+singlefalse)*100:.2f}% tokens correctly labeled")
        print(f"{singlefalse/(singlecorrect+singlefalse)*100:.2f}% tokens incorrectly labeled")


    return (correct,false) , (singlecorrect,singlefalse)
        
    