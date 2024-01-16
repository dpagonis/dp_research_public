from seabreeze.spectrometers import Spectrometer 
import matplotlib.pyplot as plt 

spec = Spectrometer.from_first_available()
l = spec.wavelengths()
i = spec.intensities()

with open('spectrum.csv', 'w') as f:
f.write('Wavelenth, Intensity\n')
for l, i in zip(wavelength, intensities):
f.write(f"{l},{i}\n")
