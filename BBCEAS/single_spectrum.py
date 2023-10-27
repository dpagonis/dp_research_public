from seabreeze.spectrometers import Spectrometer 
import matplotlib.pyplot as plt 
import numpy as np

spec = Spectrometer.from_first_available()
l = spec.wavelengths()
i = spec.intensities()

i[0:2]=np.nan

plt.scatter(l,i)
plt.show()

