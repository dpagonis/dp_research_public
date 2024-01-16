import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from seabreeze.spectrometers import Spectrometer

class SpectrumPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.spec = Spectrometer.from_first_available()

        # Set up the main window
        self.setWindowTitle("Live Spectrum Plot")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget (required by QMainWindow)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a vertical layout
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # Create a plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Create a plot item on the widget
        self.plot_data = self.plot_widget.plot([], [])

        # Timer to update plot (can be adjusted)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

    def update_plot(self):
        #get the data
        l = self.spec.wavelengths()
        i = self.spec.intensities()
        
        #put it on the window
        self.plot_data.setData(l, i)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectrumPlotter()
    window.show()
    sys.exit(app.exec_())
