class Tag():
    
    EMPTYTAG = "_"

    def __init__(self , omit_lemma = True):
        self.vsa = ""
        self.assigned = -1 
        self.delegates = []
        self.omit_lemma = omit_lemma

    def isempty(self):
        return not(self.vsa or self.assigned != -1 or self.delegates)

    def isassigned(self):
        return self.assigned != -1
    
    def isverb(self):
        return self.vsa != ""

    def __str__(self):
        s = ""
        
        if self.isempty():
            return Tag.EMPTYTAG
        
        if self.isverb() :
            if self.omit_lemma:
                s += "V" + self.vsa[-2::]
            else :
                s += self.vsa
        
        if self.isassigned():
            if self.isverb():
                s += "*" + str(self.assigned)
            else :
                s += str(self.assigned)
        
        for dlg in self.delegates:
            s += f"++{dlg[0]},{dlg[1]}"

        return s        

    
    