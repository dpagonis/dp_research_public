import sys
import numpy as np
import datetime
import csv
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from seabreeze.spectrometers import Spectrometer
import time
import subprocess
import shutil

class CombinedPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the spectrometer
        self.spec = Spectrometer.from_first_available()
        default_int_time = 100000
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
        self.intensity_plot_data = self.intensity_plot_widget.plot([], [], pen=pg.mkPen('royalblue', width=2))
        self.timestamps = []
        self.intensities = []

        # Integration time slider
        self.int_time_slider = QSlider(Qt.Horizontal)
        self.int_time_slider.setMinimum(50)
        self.int_time_slider.setMaximum(500000)
        self.int_time_slider.setValue(default_int_time)
        self.int_time_slider.valueChanged.connect(self.set_integration_time)
        self.main_layout.addWidget(self.int_time_slider)

        # Integration time label
        self.int_time_label = QLabel(f"Integration Time: {default_int_time/1e6:.02g} s")
        self.main_layout.addWidget(self.int_time_label)

        # Create a save data button
        self.save_button = QPushButton("Save Data to CSV")
        self.main_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_data)

        # Timer to update plots
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)
    
    #saves csv timestamp vs intensity
    # def save_data(self):
    #     timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    #     filename = f"data_{timestamp_str}.csv"
    #     filepath = '/home/atmoschem/Documents/CSV_DATA'
    #     full_path = os.path.join(filepath, filename)  # Combine the filepath and filename

    #     with open(full_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['Timestamp', 'Intensity'])
    #         for timestamp, intensity in zip(self.timestamps, self.intensities):
    #             writer.writerow([timestamp, intensity])
    
    #saves csv wavelength vs intensity
    def save_data(self):
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"spectrum_data_{timestamp_str}.csv"
        filepath = '/home/atmoschem/Documents/CSV_DATA'
        full_path = os.path.join(filepath, filename)  # Combine the filepath and filename

        # Capture the current spectrum data
        wavelengths = self.spec.wavelengths()
        intensities = self.spec.intensities()

        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'Intensity'])
            for wavelength, intensity in zip(wavelengths, intensities):
                writer.writerow([wavelength, intensity])

        print(f"Spectrum data saved to {full_path}")
        #print(f"Data saved to {full_path}")

    def update_plots(self):
        # Update spectrum plot
        wavelengths = self.spec.wavelengths()
        intensities = self.spec.intensities()
        if len(intensities) > 2:
            intensities[0:2] = np.nan  # Assuming intensities is a NumPy array
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