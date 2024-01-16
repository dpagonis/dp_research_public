import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QTabWidget
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from seabreeze.spectrometers import Spectrometer
import time

class IntensityPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        default_int_time = 500
        self.spec = Spectrometer.from_first_available()
        self.spec.integration_time_micros(default_int_time)

        # Set up the main window
        self.setWindowTitle("Intensity over Time (445-455 nm)")
        self.setGeometry(100, 100, 800, 600)
        self.resizable(width=False, height=False)  #set resizable window

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a vertical layout on the central widget
        layout = QVBoxLayout(self.central_widget)

        # Create a tab widget to switch between plot and table
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create a plot widget for the first tab
        self.plot_widget = pg.PlotWidget()
        self.tab_widget.addTab(self.plot_widget, "Plot")

        # Create a plot item on the widget
        self.plot_widget.setLabel('bottom', 'Time', units='s')  # x-axis
        self.plot_widget.setLabel('left', 'Intensity')  # y-axis
        self.plot_data = self.plot_widget.plot([], [], pen=pg.mkPen('w', width=2))

        self.timestamps = []
        self.intensities = []

        #Connect the mouse click event to the handler
        #A scene is the way that Seabreeze is telling the spectrometer to take measurements

        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)
            
        def on_click(self, event):
            if event.button() == 1: #left mouse button
                mouse_point = self.plot_widget.plotItem.vb.mapToView(event.pos())
                x = mouse_point.x()
                y = mouse_point.y()

        # Integration time control
        self.int_time_slider = QSlider(Qt.Horizontal)
        self.int_time_slider.setMinimum(10)
        self.int_time_slider.setMaximum(100_000)
        self.int_time_slider.setValue(default_int_time)  # Default
        self.int_time_slider.valueChanged.connect(self.set_integration_time)
        layout.addWidget(self.int_time_slider)

        # Display the current integration time
        self.int_time_label = QLabel(f"Integration Time: {default_int_time / 1e6:.02g} s")
        layout.addWidget(self.int_time_label)

        # Timer to update plot
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

        # Add a button to display the table in the second tab
        self.show_table_button = QPushButton("Show Table")
        self.show_table_button.clicked.connect(self.show_table)
        layout.addWidget(self.show_table_button)

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
        self.int_time_label.setText(f"Integration Time: {value / 1_000_000:.2f} s")

    def show_table(self):
        # Create a new QWidget for the table in the second tab
        table_widget = QWidget()
        self.tab_widget.addTab(table_widget, "Table")

        # Create a vertical layout for the table widget
        table_layout = QVBoxLayout(table_widget)

        # Create a QTableWidget
        table = QTableWidget()
        table_layout.addWidget(table)

        # Set headers for the table
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Time [sec]", "Intensity"])

        # Populate the table with data
        for row, (timestamp, intensity) in enumerate(zip(self.timestamps, self.intensities)):
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(str(timestamp)))
            table.setItem(row, 1, QTableWidgetItem(str(intensity)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntensityPlotter()
    window.show()
    sys.exit(app.exec_())