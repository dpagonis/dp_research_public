import sys
import numpy as np
import datetime
import csv
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QPushButton, QLineEdit
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
        default_int_time = 14000000
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
        self.int_time_slider.setMaximum(15000000)
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

        # Add Pause and Play Buttons
        self.pause_button = QPushButton("Pause")
        self.play_button = QPushButton("Play")
        
        # Add buttons to the main layout
        self.main_layout.addWidget(self.pause_button)
        self.main_layout.addWidget(self.play_button)

        # Connect buttons to their respective slot functions
        self.pause_button.clicked.connect(self.pause_timer)
        self.play_button.clicked.connect(self.start_timer)

        # new code for averaging time input
        # Averaging time input
        self.avg_time_input = QLineEdit()
        self.avg_time_input.setPlaceholderText("Enter averaging time (s)")
        self.main_layout.addWidget(self.avg_time_input)

        # Button to take a single spectrum
        self.take_spectrum_button = QPushButton("Take Spectrum")
        self.main_layout.addWidget(self.take_spectrum_button)
        self.take_spectrum_button.clicked.connect(self.take_spectrum)

        # Plot for single spectrum
        self.single_spectrum_plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.single_spectrum_plot_widget)
        self.single_spectrum_plot_widget.setLabel('bottom', 'Wavelength', units='nm')
        self.single_spectrum_plot_widget.setLabel('left', 'Intensity')
        self.single_spectrum_data = self.single_spectrum_plot_widget.plot([], [])
        #end of new code

        # Timer to update plots
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

        self.last_save_time = datetime.datetime.now()
        self.current_integration_time = self.int_time_slider.value()  # Get current value from the slider


    # new code for averaging time input
    def take_spectrum(self):
        try:
            averaging_time = float(self.avg_time_input.text())
        except ValueError:
            print("Invalid averaging time input")
            return
        
        # Store current integration time from the slider
        #current_int_time = self.int_time_slider.value()

        self.spec.integration_time_micros(int(averaging_time * 1e6))
        wavelengths = self.spec.wavelengths()
        intensities = self.spec.intensities()
        # Filter wavelengths below 200 nm
        mask = wavelengths >= 200
        wavelengths = wavelengths[mask]
        intensities = intensities[mask]

        self.single_spectrum_data.setData(wavelengths, intensities)
        self.save_single_spectrum(wavelengths, intensities,averaging_time)

        # Reset integration time back to slider value
        #self.spec.integration_time_micros(current_int_time)
        #self.int_time_label.setText(f"Integration Time: {current_int_time / 1e6:.02g} s")
    #end of new code

    #saves csv wavelength vs intensity to private repository
    def save_data(self):
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        averaging_time_ms = int(self.current_integration_time / 1000) #averagine time in ms
        filename = f"single_spectrum_{timestamp_str}_{averaging_time_ms}msAvTime.csv"

        # Path to the local clone of the private repository
        private_repo_path = '/home/atmoschem/software/dp_research_private/'
        private_target_path = os.path.join(private_repo_path, '2024_Whitten_BBCEAS/HeliumExperiment', filename)

        # Create directories if they do not exist
        os.makedirs(os.path.dirname(private_target_path), exist_ok=True)

        # Save spectrum data directly to the private repository
        wavelengths = self.spec.wavelengths()
        intensities = self.spec.intensities()

        with open(private_target_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'Intensity'])
            for wavelength, intensity in zip(wavelengths, intensities):
                writer.writerow([wavelength, intensity])

        print(f"Spectrum data saved to {private_target_path}")

        # Git operations to add, commit, and push the file to the private repository
        try:
            os.chdir(private_repo_path)
            subprocess.run(['git', 'add', private_target_path], check=True)
            subprocess.run(['git', 'commit', '-m', f'Add new spectrum data: {filename}'], check=True)
            subprocess.run(['git', 'push'], check=True)
            print(f"File {filename} pushed to private GitHub repository successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while pushing to GitHub: {e}")

    #new code for averaging time --> saves the averaged time spectrum data
    def save_single_spectrum(self, wavelengths, intensities, averaging_time):
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        averaging_time_ms = int(averaging_time * 1000) #averagine time in ms
        filename = f"single_spectrum_{timestamp_str}_{averaging_time_ms}msAvTime.csv"

        # Path to the local clone of the private repository
        private_repo_path = '/home/atmoschem/software/dp_research_private/'
        private_target_path = os.path.join(private_repo_path, '2024_Whitten_BBCEAS/HeliumExperiment', filename)

        with open(private_target_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'Intensity'])
            for wavelength, intensity in zip(wavelengths, intensities):
                    writer.writerow([wavelength, intensity])

        print(f"Single spectrum data saved to {private_target_path}")

        # Git operations to add, commit, and push the file to the private repository
        try:
            os.chdir(private_repo_path)
            subprocess.run(['git', 'add', private_target_path], check=True)
            subprocess.run(['git', 'commit', '-m', f'Add new single spectrum data: {filename}'], check=True)
            subprocess.run(['git', 'push'], check=True)
            print(f"File {filename} pushed to private GitHub repository successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while pushing to GitHub: {e}")

    #end of new code

    def update_plots(self):
        # Update spectrum plot
        wavelengths = self.spec.wavelengths()
        intensities = self.spec.intensities()
        # Filter wavelengths below 200 nm
        mask = wavelengths >= 200
        wavelengths = wavelengths[mask]
        intensities = intensities[mask]

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

         # Determine if it's time to save again (e.g., every 60 seconds)
        current_time = datetime.datetime.now()
        if (current_time - self.last_save_time).total_seconds() > 30:
            self.current_integration_time = self.int_time_slider.value()  # Get current value from the slider
            self.save_data()  # Pass this value to save_data
            self.last_save_time = current_time

    def set_integration_time(self, value):
        self.spec.integration_time_micros(value)
        self.int_time_label.setText(f"Integration Time: {value / 1_000_000:.2f}s")

    def pause_timer(self):
        """Slot function to pause the timer."""
        self.timer.stop()
        print("Timer paused")

    def start_timer(self):
        """Slot function to start the timer."""
        self.timer.start(100)  # You may adjust the interval as needed
        print("Timer started")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CombinedPlotter()
    window.show()
    sys.exit(app.exec_())