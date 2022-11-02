from sklearn import linear_model
import pandas as pd
import numpy as np
import scipy.stats

class PB_Design:

    sDesigns = {
        4 : "++-",
        8 : "+++-+--",
        12:"++-+++---+-",
        16:"++++-+-++--+---",
        20:"++--++++-+-+----++-",
        24:"+++++-+-++--++--+-+----"
    }

    def __init__(self,nX=None,nY=None,center=None):
     #define number of independent variables
        if nX is None:
            self.nX = 2
        else:
            self.nX=nX

        #define number of independent variables
        if nY is None:
            self.nY = 1
        else:
            self.nY=nY

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
        self.Design_plusminus = pd.DataFrame(np.transpose(CharArray))
        self.Design_plusminus.columns = [f"x{i}" for i in self.Design.columns]
        self.Design_plusminus.index = [f"r{i}" for i in self.Design.index]
        self.Design.columns = [f"x{i}" for i in self.Design.columns]
        self.Design.index = [f"r{i}" for i in self.Design.index]
        self.Design[self.Design=='-'] = 0
        self.Design[self.Design=='+'] = 1
        
        if center is True:
          self.Design.loc['r'+str(self.nReplicates)] = 0.5
          self.Design_plusminus.loc['r'+str(self.nReplicates)] = '0'
          self.nReplicates += 1
          self.curvature = None
          self.center = True
        
        #define all parameters with number of rows = nReplicates, then make yData dataframe for all of those results
          # y, y_std, y_N
        self.y = [f"y{i}" for i in range(self.nY)]
        self.y_std = [f"y{i}_std" for i in range(self.nY)]
        self.y_n=[f"y{i}_n" for i in range(self.nY)]
        self.yData=pd.DataFrame(index=(i for i in range(self.nReplicates)),columns=self.y+self.y_std+self.y_n)
        self.yData_sPooled = pd.DataFrame(index=['s_pooled','crit'],columns=self.y)
        self.xCoefs = pd.DataFrame(index=self.Design.columns,columns=self.y)
        self.significance = pd.DataFrame(index=self.Design.columns,columns=self.y)

    
    def LoadData(self,sDataDir):
      data = pd.read_csv(sDataDir)
      
      #calculate averages, std deviations, n replicates at each run
      for i in range(self.nReplicates):
        for j in range(self.nY):
          c = self.y[j]
          d = self.y_std[j]
          e = self.y_n[j]
          self.yData.at[i,c] = data[data['run']==i][c].mean()
          self.yData.at[i,d] = data[data['run']==i][c].std()
          self.yData.at[i,e] = len(data[data['run']==i][c])
      
      #calculate s_pooled
      for c in self.yData_sPooled.columns:
        num = 0
        denom = 0
        for i in range(self.nReplicates):
          num += self.yData.at[i,c+"_std"]**2 * (self.yData.at[i,c+"_n"] - 1)
          denom += self.yData.at[i,c+"_n"] - 1
        self.yData_sPooled.at['s_pooled',c] = (num/denom)**0.5
        self.yData_sPooled.at['crit',c] = scipy.stats.t.ppf(q=1-0.05/2,df=len(data.index)-2) * (num/denom)**0.5 * (2/(self.Design['x0'].sum()*self.yData[c+"_n"].mean()))**0.5

      # do regression
      for c in self.y:
        reg = linear_model.LinearRegression()
        reg.fit(self.Design,self.yData[c])
          
        for i in range(len(reg.coef_)):
          x=self.Design.columns[i]
          self.xCoefs.at[x,c]= reg.coef_[i]
          self.significance.at[x,c] = True if abs(self.xCoefs.at[x,c]) > self.yData_sPooled.at['crit',c] else False
      print("Significance:")
      print(self.significance)

      #check curvature if centered
      if self.center is True:
        self.curvature = pd.DataFrame(index=[0],columns=self.y)
        for c in self.y:
          nCenter = self.yData.at[self.nReplicates-1,c+"_n"]
          effect = self.yData[c].sum()/nCenter
          effect_crit = scipy.stats.t.ppf(q=1-0.05/2,df=len(data.index)-2) * self.yData_sPooled.at['s_pooled',c] * (1/((self.Design['x0'].sum()-0.5)*self.yData[c+"_n"].mean())+1/nCenter)**0.5#(=ts(1/mk+1/C)^1/2)
          self.curvature.at[0,c] = True if effect>effect_crit else False
        print("")
        print("Curvature:")
        print(self.curvature)
