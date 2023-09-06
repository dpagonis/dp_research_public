import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO

EARTHRADIUS = 6.371e6 #m
SUTHERLAND = 110 #K

# Functions-----------------------------------------------------------------------------------------
# DP_DistFromLL(Coord1,Coord2): return distance in m given tuples with LAT,LON in degrees
# DP_Kn(Temp,Pres_mbar,diam_m): Knudsen number
# DP_MeanFP(Temp,Pres_mbar): input T (K) and P (mbar) return mean free path of air in m
# DP_TUV_ActinicFlux(inputs): return dataframe with actinic flux from NCAR's TUV calculator across wavelengths

# Classes-------------------------------------------------------------------------------------------
# VBS_May13 : May et al 2013 JGRA VBS



#-------------------------------------------------------------------
# ----------------------FUNCTIONS-----------------------------------
#-------------------------------------------------------------------


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


def DP_TUV_ActinicFlux(latitude, longitude, date, timeStamp, mAltitude):
    """
    Fetches UV radiation data from the NCAR TUV calculator and returns it as a DataFrame.
    
    Parameters:
    -----------
    latitude : float
        Latitude for the calculation. Should be a float between -90 and 90.
        
    longitude : float
        Longitude for the calculation. Should be a float between -180 and 180.
        
    date : str
        Date for the calculation in 'YYYYMMDD' format.
        
    timeStamp : str
        Time for the calculation in 'HH:MM:SS' format.
        
    mAltitude : float
        Altitude in meters for the measurement point. Should be a non-negative float.
        
    Returns:
    --------
    pd.DataFrame or None
        A DataFrame containing the UV radiation data if the API call is successful.
        Returns None if the API call fails.
        
    Example:
    --------
    >>> df = call_ncar_tuv(latitude=0, longitude=0, date='20150630', timeStamp='12:00:00', mAltitude=0)
    >>> print(df)
    
    Notes:
    ------
    - The function assumes that certain other parameters (e.g., 'wStart', 'ozone', etc.) are set to default values.
    - Ensure that you have access to the internet and that the NCAR TUV calculator is online and operational.
    """
    
    base_url = "https://www.acom.ucar.edu/cgi-bin/acom/TUV/V5.3/tuv"
    
    params = {
        'wStart': 280,
        'wStop': 999,
        'wIntervals': 719,
        'inputMode': 0,
        'latitude': latitude,
        'longitude': longitude,
        'date': date,
        'timeStamp': timeStamp,
        'ozone': 300,
        'zenith':0,
        'albedo': 0.1,
        'gAltitude': 0,
        'mAltitude': mAltitude,
        'taucld': 0.00,
        'zbase': 4.0,
        'ztop': 5.0,
        'tauaer': 0.0,
        'ssaaer': 0.990,
        'alpha': 1,
        'outputMode': 4,
        'nStreams': -2,
        'time':12,
        'dirsun': 1.0,
        'difdn': 1.0,
        'difup': 1.0
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        content = response.text
        print(content)  # Add this line for debugging
        data_str = content.split("\n")[23:]  # Skip first 23 lines
        data_str = "\n".join(data_str)
        
        column_names = ["LOWER WVL", "UPPER WVL", "DIRECT", "DIFFUSE DOWN", "DIFFUSE UP", "TOTAL"]
        
        df = pd.read_csv(StringIO(data_str), header=None, delim_whitespace=True, names=column_names)
        
        return df
    else:
        print(f"Failed to get data: {response.status_code}")
        return None

#-------------------------------------------------------------------
# ----------------------CLASSES-------------------------------------
#-------------------------------------------------------------------


class VBS:
  #This class is **always** at equilibirum
  #initialize with Temp (K), Pressure (mbar), and Total mass concentration of VBS (ug sm-3)
  #outputs: equilibrium OA concentrations at self.OA and self.OA_vol (ug sm-3 and ug m-3, respectively)
  #if you know OA but not total VBS conc, use self.SetOAConc(OA_std) to automatically find the total VBS conc.

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

  def Plot(self):
      plt.bar(range(len(self.CStar_298)),self.VBSMass_vol,tick_label=[str(cs) for cs in self.CStar_298])
      OAbars = self.Fp * self.VBSMass_vol
      plt.bar(range(len(self.CStar_298)),OAbars)
      plt.xlabel('C*298 (ug m-3)')
      plt.ylabel('Concentration (ug m-3)')
      plt.figtext(0.2,0.9,'VBS = %s\nVBS = %.02f ug sm-3\nOA = %.02f ug sm-3' % (self.Name,self.VBSConc, self.OA))
      plt.figtext(0.6,0.9,'T = %.01f K\nP = %.0f mbar' % (self.Temp,self.Pres))
      plt.show()

  def __init__(self,Temp=None,Pres=None,VBSConc=None):
    self.Temp = 298.15 if Temp is None else Temp
    self.Pres = 1013.25 if Pres is None else Pres
    self.VBSConc = 100 if VBSConc is None else VBSConc #ugsm3
    self.CStar = [self.Clausius_Clapeyron(cs , self.Temp , dH) for cs,dH in zip(self.CStar_298,self.dHvap)]
    self.OA = self.VBSConc / 2
    self.PartitionVBS()
#end of class VBS

class VBS_May13(VBS):
  #VBS defined in May et al JGRA 2013 (https://doi.org/10.1002/jgrd.50828)
  Name = "May_2013_JGRA"
  Fi = np.array([0.2,0,0.1,0.1,0.2,0.1,0.3])
  CStar_298 = np.array([10**(i-2) for i in range(7)])
  dHvap = np.array([(85-4*math.log10(x))*1000 for x in CStar_298])
#end of class VBS_May13

class VBS_FIREX(VBS):
    #VBS fitted to FIREX-AQ thermal denuder data
    Name = "FIREX-AQ"
    Fi = np.array([0.2,0.1,0.2,0.1,0,0.2,0.2])
    CStar_298 = np.array([0.01,0.1,1,10,100,1000,10000])
    dHvap = np.array([(85-4*math.log10(x))*1000 for x in CStar_298])
  #end of class VBS_FIREX

