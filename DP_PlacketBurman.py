import pandas as pd
import numpy as np

class PB_Design:

    sDesigns = {
        4 : "++-",
        8 : "+++-+--",
        12:"++-+++---+-",
        16:"++++-+-++--+---",
        20:"++--++++-+-+----++-",
        24:"+++++-+-++--++--+-+----"
    }

    def __init__(self,nX=None):
        #define number of independent variables
        if nX is None:
            self.nX = 2
        else:
            self.nX=nX

        #space for result arrays
        self.YVals = []

        #calculate number of replicates in PB design. Always a multiple of 4
        self.nReplicates = 4 * (self.nX//4+1)
        if self.nReplicates > 24:
            raise Exception("Sorry, class PB_Design can't go above 23 factors")

        CharArray=[]
        for i in range(self.nX):
            xi = ['-']
            k=i
            for j in range(self.nReplicates-1):
                sDesign = self.sDesigns[self.nReplicates]
                xi.append(sDesign[k])
                k = (k+1) % len(sDesign)
            CharArray.append(xi)

        #dataframe for design
        self.Design = pd.DataFrame(np.transpose(CharArray))
        self.Design.columns = [f"x{i}" for i in self.Design.columns]
        self.Design.index = [f"r{i}" for i in self.Design.index]

        #dataframe for results
        # self.YVals=pd.DataFrame(np.full([self.nReplicates,2*self.nY],None))
        # self.YVals.columns = [f"y{i//2}" if i%2 == 0 else f"sy{i//2}" for i in self.YVals.columns]
        # self.YVals.index = [f"r{i}" for i in self.YVals.index]
