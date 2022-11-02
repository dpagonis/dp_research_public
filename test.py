from DP_GenTools import *

V=VBS_May13()

plt.bar(range(len(V.CStar_298)),V.VBSMass_vol,tick_label=[str(cs) for cs in V.CStar_298])
OAbars = V.Fp * V.VBSMass_vol
plt.bar(range(len(V.CStar_298)),OAbars)
plt.xlabel('C*298 (ug m-3)')
plt.ylabel('Concentration (ug m-3)')
plt.figtext(0.3,0.8,'VBS = %.02f ug sm-3\nOA = %.02f ug sm-3' % (V.VBSConc, V.OA))
plt.show()
