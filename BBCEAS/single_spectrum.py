from seabreeze.spectrometers import Spectrometer 
import matplotlib.pyplot as plt 

spec = Spectrometer.from_first_available()
l = spec.wavelengths()
i = spec.intensities()

plt.plot(l,i,'k')

plt.show()

