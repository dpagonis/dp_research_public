import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from seabreeze.spectrometers import Spectrometer
import time

class CombinedPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the spectrometer
        self.spec = Spectrometer.from_first_available()
        default_int_time = 5000
        self.spec.integration_time_micros(default_int_time)

        # Set up the main window
        self.setWindowTitle("Spectrometer Analysis")
        self.setGeometry(100, 100, 1200, 600)

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main vertical layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create a horizontal layout for the plots
        self.plot_layout = QHBoxLayout()
        self.main_layout.addLayout(self.plot_layout)

        # Create the live spectrum plot
        self.plot_widget = pg.PlotWidget()
        self.plot_layout.addWidget(self.plot_widget)
        self.plot_widget.setLabel('bottom', 'Wavelength', units='nm')
        self.plot_widget.setLabel('left', 'Intensity')
        self.plot_widget.setXRange(200, 550)
        self.plot_data = self.plot_widget.plot([], [])

        # Create the intensity plot
        self.intensity_plot_widget = pg.PlotWidget()
        self.plot_layout.addWidget(self.intensity_plot_widget)

        self.intensity_plot_widget.setLabel('bottom', 'Time', units='s')
        self.intensity_plot_widget.setLabel('left', 'Intensity')
        self.plot_widget.setXRange(400, 550)
        self.intensity_plot_data = self.intensity_plot_widget.plot([], [], pen=pg.mkPen('royalblue', width=2))
        self.timestamps = []
        self.intensities = []

        # Integration time slider
        self.int_time_slider = QSlider(Qt.Horizontal)
        self.int_time_slider.setMinimum(50)
                                    
        self.int_time_slider.setMaximum(500_000)-
        self.int_time_slider.setValue(default_int_time)
        self.int_time_slider.valueChanged.connect(self.set_integration_time)
        self.main_layout.addWidget(self.int_time_slider)

        # Integration time label
        self.int_time_label = QLabel(f"Integration Time: {default_int_time/1e6:.02g} s")
        self.main_layout.addWidget(self.int_time_label)

        # Timer to update plots
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

    def update_plots(self):
        # Update spectrum plot
        wavelengths = self.spec.wavelengths()
        intensities = self.spec.intensities()
        intensities[0:2]=np.nan
        self.plot_data.setData(wavelengths, intensities)

        # Update intensity plot
        mask = (wavelengths >= 445) & (wavelengths <= 455)
        avg_intensity = np.mean(intensities[mask])
        self.timestamps.append(time.time())
        self.intensities.append(avg_intensity)
        self.intensity_plot_data.setData(self.timestamps, self.intensities)
        self.intensity_plot_widget.setXRange(self.timestamps[-1] - 60, self.timestamps[-1])

    def set_integration_time(self, value):
        self.spec.integration_time_micros(value)
        self.int_time_label.setText(f"Integration Time: {value / 1_000_000:.2f}s")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CombinedPlotter()
    window.show()
    sys.exit(app.exec_())
