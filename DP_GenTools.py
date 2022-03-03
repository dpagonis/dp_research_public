import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

EARTHRADIUS = 6.371e6 #m
SUTHERLAND = 110 #K

# Functions-----------------------------------------------------------------------------------------
# DP_DistFromLL(Coord1,Coord2): return distance in m given tuples with LAT,LON in degrees
# DP_Kn(Temp,Pres_mbar,diam_m): Knudsen number
# DP_MeanFP(Temp,Pres_mbar): input T (K) and P (mbar) return mean free path of air in m

# Classes-------------------------------------------------------------------------------------------
# VBS_May13 : May et al 2013 JGRA VBS

def DP_DistFromLL(Coord1,Coord2):
    #return distance in m given tuples with LAT,LON in degrees


    Lat1 = Coord1[0] * math.pi / 180 #in radians
    Lon1 = Coord1[1] * math.pi / 180
    Lat2 = Coord2[0] * math.pi / 180
    Lon2 = Coord2[1] * math.pi / 180

    dist_rad = 2 * math.asin(math.sqrt((math.sin((Lat1-Lat2)/2))**2 + math.cos(Lat1)*math.cos(Lat2)*(math.sin((Lon1-Lon2)/2))**2))
    dist_m = dist_rad * EARTHRADIUS

    return dist_m

def DP_Kn(Temp,Pres_mbar,diam_m):
    #input T (K), P (mbar), diameter (m)
    #return Knudsen number
    return 2 * DP_MeanFP(Temp,Pres_mbar) / diam_m

def DP_MeanFP(Temp,Pres_mbar):
    #input T (K) and P (mbar)
    #return mean free path of air in m
    return 6.65e-8 * (1013.25/Pres_mbar) * (Temp / 273.15) * (1+SUTHERLAND/273.15) / (1+SUTHERLAND/Temp)


#-------------------------------------------------------------------
# ----------------------CLASSES-------------------------------------
#-------------------------------------------------------------------

class VBS_May13:
  #VBS defined in May et al JGRA 2013 (https://doi.org/10.1002/jgrd.50828)
  #This class is **always** at equilibirum
  #initialize with Temp (K), Pressure (mbar), and Total mass concentration of VBS (ug sm-3)
  #outputs: equilibrium OA concentrations at self.OA and self.OA_vol (ug sm-3 and ug m-3, respectively)
  #if you know OA but not total VBS conc, use self.SetOAConc(OA_std) to automatically find the total VBS conc.

  Fi = np.array([0.2,0,0.1,0.1,0.2,0.1,0.3])
  CStar_298 = np.array([10**(i-2) for i in range(7)])
  dHvap = np.array([(85-4*math.log10(x))*1000 for x in CStar_298])

  def Clausius_Clapeyron(self, Cstar298 , T , dH_298):
    Cstar = ((Cstar298*298.15)/T)*math.e**((dH_298/8.3145)*((1/298.15)-(1/T)))
    return Cstar

  def SetTP(self,T=None,P=None):                # set temperature and pressure conditions for VBS
    self.Temp = self.Temp if T is None else T
    self.Pres = self.Pres if P is None else P
    self.CStar = [self.Clausius_Clapeyron(cs , self.Temp , dH) for cs,dH in zip(self.CStar_298,self.dHvap)]
    self.PartitionVBS()

  def CalcVolConcs(self):
    self.StdToVol = (273.15/self.Temp) * (self.Pres/1013.25) #Std * StdToVol = Vol ; ATTN: this is flipped from the Jimenez group convention
    self.VBSConc_vol = self.VBSConc * self.StdToVol
    self.VBSMass_vol = self.VBSConc_vol * self.Fi
    self.OA_vol = self.OA * self.StdToVol
    #standard concentration floats: VBSConc, OA
    #volumetric conctration floats: VBSConc_vol, OA_vol
    #volumetric concntration array: VBSMass_vol

  def SetVBSConc(self,VBSConc_std): #directly set the total VBS concentration (ug sm-3)
    self.VBSConc = VBSConc_std
    self.PartitionVBS()

  def SetOAConc(self,OA_std):   #set the total VBS concentration by inputting the equilibrium-partitioned OA concentration (ug sm-3)
    while abs(self.OA-OA_std)/OA_std > 1e-6:
      self.VBSConc = self.VBSConc * OA_std / self.OA
      self.PartitionVBS()

  def ScaleConc(self,ScaleFactor):  #scale the total VBS concentration. e.g. ScaleFactor = 0.9 represents 10% dilution
    self.VBSConc = self.VBSConc * ScaleFactor
    self.PartitionVBS()

  def PartitionVBS(self):
    self.CalcVolConcs()
    OACalc = self.OA_vol
    dOA = 1
    while dOA>1e-6:
      Fp = [(1 + (x/OACalc))**-1 for x in self.CStar]
      Mass_p = self.VBSMass_vol * Fp
      LastCOA = OACalc
      OACalc = np.sum(Mass_p)
      dOA = (abs(LastCOA - OACalc)/OACalc)
    self.Fp = Fp
    self.OA_vol = OACalc
    self.OA = self.OA_vol / self.StdToVol

  def __init__(self,Temp=None,Pres=None,VBSConc=None):
    self.Temp = 298.15 if Temp is None else Temp
    self.Pres = 1013.25 if Pres is None else Pres
    self.VBSConc = 100 if VBSConc is None else VBSConc #ugsm3
    self.CStar = [self.Clausius_Clapeyron(cs , self.Temp , dH) for cs,dH in zip(self.CStar_298,self.dHvap)]
    self.OA = self.VBSConc / 2
    self.PartitionVBS()
#end of class VBS_May13
