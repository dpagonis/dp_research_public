import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint

class Inlet:
  #class that contains the temperature and OA evaporation info for an inlet. Aircraft-specific features,...
  #....can be used for ground inlets by setting TATTime=0
  #initialize with Ambient Temp (K),Inlet Temp (K), Total air temp (TAT) (K),Pressure (mbar), Tube diameter (m),... 
  #....Total residence time (s) , Time spent inside inlet but outside plane (TATTime) (s)
  #
  #InletTemp vs InletTime is the heating curve. calculated by simple heat transfer
  #
  #CalcOAEvap(VBS) takes a VBS class and gives the evap curve InletOA vs InletTime
  #InletOA in volumetric concentration
  #Also calculates the equilibrium curve InletOA_Equil vs InletTime to compare to the Cappa kinetic model 
  #
  #Pre-programmed graphs:
  # - Inlet temp vs time           :  PlotTemp()
  # - InletOA vs time              :  PlotOA()
  # - InletOA_equil vs time        :  PlotOAEquil()
  # - evap curves for each VBS bin :  PlotVBSEvap()

  def SetTP(self,AmbientT=None,TAT=None,InletT=None,Pres=None):
    self.AmbientT = self.AmbientT if AmbientT is None else AmbientT
    self.TAT = self.AmbientT+10 if TAT is None else TAT
    self.InletT = self.InletT if InletT is None else InletT
    self.Pres = self.Pres if Pres is None else Pres
    self.CalcTemps()

  def SetResTime(self,ResTime=None,TATTime=None):
    self.ResTime = self.ResTime if ResTime is None else ResTime
    self.TATTime = self.TATTime if TATTime is None else TATTime 
    self.InletTime = np.array([x/100 for x in range(round(self.ResTime*100))])
    self.CalcTemps()
    self.InletOA = np.zeros(len(self.InletTime))
    self.InletOA_Equil = np.zeros(len(self.InletTime))
    self.InletMFR = np.zeros(len(self.InletTime))

  def SetdTube(self,dTube):
    self.dTube=dTube
    self.CalcTemps()

  def CalcTemps(self):
    self.InletTemp = odeint(DP_Inlet_dTdt,self.AmbientT,self.InletTime,args=(self,))

  def CalcOAEvap(self,VBS):
    #hardcoded Dp
    Dp_nm = 300
    ###
    self.VBS = VBS
    OAVBS_vol = VBS.Fp * VBS.VBSMass_vol
    self.InletOA_VBS = odeint(DP_Inlet_dOAdt,OAVBS_vol,self.InletTime,args=(self,VBS,Dp_nm))
    self.InletOA = [np.sum(x) for x in self.InletOA_VBS]
    self.InletMFR = self.InletOA/self.InletOA[0]
    self.OAMFR = self.InletOA[-1]/self.InletOA[0]
    self.CalcOAEvapEquilibrium(VBS)
    
  def CalcOAEvapEquilibrium(self,VBS):
    self.VBS = VBS
    T = VBS.Temp
    for i in range(len(self.InletTime)):
      self.VBS.SetTP(T=float(self.InletTemp[i]))
      self.InletOA_Equil[i] = self.VBS.OA_vol
    self.VBS.SetTP(float(T))

  def PlotTemp(self):
    plt.plot(self.InletTime,self.InletTemp)
    plt.xlabel('Time(s)')
    plt.ylabel('Temperature(K)')
    plt.show()

  def PlotOA(self):
    plt.plot(self.InletTime,self.InletOA)
    plt.xlabel('Time(s)')
    plt.ylabel('OA (ug m3)')
    plt.ylim(bottom=0)
    plt.show()

  def PlotOAEquil(self):
    plt.plot(self.InletTime,self.InletOA)
    plt.plot(self.InletTime,self.InletOA_Equil)
    plt.xlabel('Time(s)')
    plt.ylabel('OA (ug m3)')
    plt.ylim(bottom=0)
    plt.legend(['InletOA', 'InletOA_Equil'])
    plt.show()

  def PlotVBSEvap(self):
    n = len(self.VBS.CStar)
    colors = plt.cm.cividis(np.linspace(0,1,n))
    for i in range(n):  
      plt.plot(self.InletTime,self.InletOA_VBS[:,i],linewidth=3, color=colors[i])
    plt.legend([str(cs) for cs in self.VBS.CStar_298])
    plt.xlabel('Time(s)')
    plt.ylabel('Concentration (ug m3)')
    plt.show()

  def PlotMFR(self):
    plt.plot(self.InletTime,self.InletMFR)
    plt.xlabel('Time(s)')
    plt.ylabel('MFR')
    plt.ylim(0,1)
    plt.show()
    
  def __init__(self,AmbientT=None,InletT=None,TAT=None,Pres=None,dTube=None,ResTime=None,TATTime=None):
    
    #set defaults
    self.AmbientT=273.15 if AmbientT is None else AmbientT
    self.InletT=298.15 if InletT is None else InletT
    self.TAT= self.AmbientT+10 if TAT is None else TAT
    self.Pres = 1013.25 if Pres is None else Pres
    self.dTube = 4.7e-3 if dTube is None else dTube #1/4" ID tube
    self.ResTime = 3 if ResTime is None else ResTime
    self.TATTime = 0.05 if TATTime is None else TATTime
    self.OAMFR = None
    self.VBS = None

    #make arrays
    self.InletTime = np.array([x/100 for x in range(round(self.ResTime*100))])
    self.InletTemp = np.zeros(len(self.InletTime))
    self.InletOA = np.zeros(len(self.InletTime))
    self.InletMFR = np.zeros(len(self.InletTime))
    self.InletOA_Equil = np.zeros(len(self.InletTime))
    self.CalcTemps()
#end of class 'Inlet'

def DP_Inlet_dOAdt(OA,t,Inlet,VBS,Dp_nm): #calculate dOAdtime given the Inlet,VBS,and particle diameter. OA is an array with mass in particle phase for each VBS bin
  #Cappa 2010 AMT formulation, but with no radial dependence doi:10.5194/amt-3-579-2010
  timeindex = np.searchsorted(Inlet.InletTime,t) 
  timeindex = timeindex if timeindex < len(Inlet.InletTime) else timeindex-1
  Temp = float(Inlet.InletTemp[timeindex])
  Pres = Inlet.Pres
  CStar_298 = VBS.CStar_298
  dHvap = VBS.dHvap
  Gas = VBS.VBSMass_vol - OA

  Dg = 0.067e-4 #m2/s diffusion coef
  Kn = DP_Kn(Temp,Pres,Dp_nm/1e9)
  CappaGamma = (0.75+Kn*0.75)/(Kn**2+Kn+0.283*Kn+0.75)
  particlemass = 4/3*3.14159*((Dp_nm/1e9)/2)**3*100**3*1e6 #ug, assuming unit density
  Np = np.sum(OA)/particlemass
  CStar = [((x*298.15)/Temp)*math.e**((y/8.3145)*((1/298.15)-(1/Temp))) for x,y in zip(CStar_298,dHvap)]
  molfrac = [x/np.sum(OA) for x in OA]
 
  dOAdt_VBS = [Np*2*3.14159*Dg*(Dp_nm/1e9)*CappaGamma*(cg - cs*x) for cg,cs,x in zip(Gas,CStar,molfrac)]
  return dOAdt_VBS

def DP_Inlet_dTdt(Temp,t,Inlet): #calculate dtemp dtime given an input temp, time, and inlet class
  TubeT = Inlet.TAT if t < Inlet.TATTime else Inlet.InletT
  volume = (math.pi * ((Inlet.dTube/2)**2) * 0.01)
  SA = math.pi * Inlet.dTube * 0.01
  h = 0.3942 + (0.1222 * (Inlet.Pres*100)**0.37378)
  Cp = 1005 #J/kgK
  density = 1.2922 * (273.15/Temp) * (Inlet.Pres/1013.25) #kg/m3
  m = density*volume
  return (h * SA * (TubeT-Temp))/(Cp*m) 

def DP_Kn(Temp,Pres_mbar,diam_m):
    #input T (K), P (mbar), diameter (m)
    #return Knudsen number
    return 2 * DP_MeanFP(Temp,Pres_mbar) / diam_m

def DP_MeanFP(Temp,Pres_mbar):
    #input T (K) and P (mbar)
    #return mean free path of air in m
    return 6.65e-8 * (1013.25/Pres_mbar) * (Temp / 273.15) * (1+110/273.15) / (1+110/Temp)
