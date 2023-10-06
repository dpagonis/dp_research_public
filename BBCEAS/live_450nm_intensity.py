import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QSlider, QVBoxLayout, QLabel, QSpinBox
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from seabreeze.spectrometers import Spectrometer
import time
 
class IntensityPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        default_int_time = 100
        self.spec = Spectrometer.from_first_available()
        self.spec.integration_time_micros(default_int_time)

        # Set up the main window
        self.setWindowTitle("Intensity over Time (445-455 nm)")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a vertical layout
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # Create a plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Create a plot item on the widget
        self.plot_widget.setLabel('bottom', 'Time', units='s')  # x-axis
        self.plot_widget.setLabel('left', 'Intensity')  # y-axis
        self.plot_data = self.plot_widget.plot([], [], pen=pg.mkPen('w', width=2))

        self.timestamps = []
        self.intensities = []

        # Integration time control 
        
        self.int_time_slider = QSlider(Qt.Horizontal)
        self.int_time_slider.setMinimum(10)
        self.int_time_slider.setMaximum(100_000)
        self.int_time_slider.setValue(default_int_time)  # Default 
        self.int_time_slider.valueChanged.connect(self.set_integration_time)
        layout.addWidget(self.int_time_slider)

        # Display the current integration time
        self.int_time_label = QLabel(f"Integration Time: {default_int_time/1e6:.02g} s")
        layout.addWidget(self.int_time_label)

        # Timer to update plot 
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

    def update_plot(self):
        l = self.spec.wavelengths()
        i = self.spec.intensities()

        # Get average intensity for the range 445-455 nm
        mask = (l >= 445) & (l <= 455)
        avg_intensity = np.mean(i[mask])

        # Append to data
        self.timestamps.append(time.time())
        self.intensities.append(avg_intensity)

        # Update the plot
        self.plot_data.setData(self.timestamps, self.intensities)

        # Optional: Adjust x-axis to show recent data if you don't want to display all data points
        # e.g., display last 60 seconds of data
        self.plot_widget.setXRange(self.timestamps[-1] - 60, self.timestamps[-1])

    def set_integration_time(self, value):
        # Set spectrometer integration time
        self.spec.integration_time_micros(value)

        # Update label to display the current integration time
        self.int_time_label.setText(f"Integration Time: {value / 1_000_000:.2f}s")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntensityPlotter()
    window.show()
    sys.exit(app.exec_())
