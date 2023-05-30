import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def AdjGuess(wG, wE, NSmooth): # Subtract error from guess
    if NSmooth == 0 or NSmooth == 1:
        wG = wG.astype(float)  # Convert wG to float data type
        wG -= wE
    elif NSmooth > 1:
        wE_Smooth = np.convolve(wE, np.ones(NSmooth) / NSmooth, mode='same')
        wG = wG.astype(float)  # Convert wG to float data type
        wG -= wE_Smooth
    else:
        print("Error (DP_AdjGuess in DP_Deconvolution): NSmooth =", NSmooth)
        raise ValueError("DP_AdjGuess in DP_Deconvolution was passed a bad value for NSmooth")
    
    wG = np.where(wG < 0, 0, wG)
    return wG

def DP_DblExp_NormalizedIRF(x, A1, tau1, tau2):
    return A1 * np.exp(-x / tau1) + (1 - A1) * np.exp(-x / tau2)

def DP_FitDblExp(wY, wX, PtA=None, PtB=None, x0=None, x1=None, y0=None, y1=None, A1=None, tau1=None, tau2=None):
    wX = np.array(wX)  # Convert wX to a numpy array
    wY = np.array(wY)  # Convert wY to a numpy array

    if PtA is None:
        PtA = 0

    if PtB is None:
        PtB = len(wX) - 1

    if x0 is None:
        PtA = 0
    else:
        PtA = int(np.ceil(np.interp(x0, wX, np.arange(len(wX)))))

    if x1 is not None:
        PtB = int(np.interp(x1, wX, np.arange(len(wX))))

    if y0 is None:
        y0 = wY[PtB]

    NormFactor = wY[PtA]  # Store the normalization factor

    if y1 is not None:
        NormFactor = y1

    if A1 is None:
        A1 = 0.5

    if tau1 is None:
        tau1 = 1

    if tau2 is None:
        tau2 = 80

    # Extract the required portion of wY
    wY = wY[PtA:PtB+1]

    # Normalize the wY data
    wY_norm = np.where((NormFactor - y0) != 0, (wY - y0) / (NormFactor - y0), np.nan)

    #set x offset of data
    x0 = wX[PtA]
    # Fit the double exponential curve
    p0 = [A1, tau1, tau2]
    popt, pcov = curve_fit(DP_DblExp_NormalizedIRF, wX[PtA:PtB+1] - x0, wY_norm, p0=p0, bounds=([0,0,0],[1,3600,3600]))

    # Generate the fitted curve
    fitX = wX[PtA:PtB+1]
    fitY = DP_DblExp_NormalizedIRF(fitX - x0, *popt) * (NormFactor - y0) + y0
    
    return popt, pcov, fitX, fitY

if __name__ == "__main__":
    # Read data from CSV file using pandas
    csv_file_path = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/2019_08_07_HNO3Data.csv'
    data = pd.read_csv(csv_file_path)

    # Define the indices or range of the subset of data to plot and fit
    start_index = 2107
    end_index = 2407

    # Extract the subset of data to plot and fit
    x_csv_subset = data['time'][start_index:end_index]  # Replace 'x_column' with the actual column name for x values
    y_csv_subset = data['HNO3_191_Hz'][start_index:end_index]  # Replace 'y_column' with the actual column name for y values

    # Call the fitting function with the subset of data
    fitted_params, covariance, fitX, fitY = DP_FitDblExp(y_csv_subset, x_csv_subset)
    print(fitted_params)
    # Generate x values for plotting the curve
    x_plot = np.linspace(min(x_csv_subset), max(x_csv_subset), 100)

    # Evaluate the fitted double exponential function at x_plot
    y_plot = DP_DblExp_NormalizedIRF(x_plot, *fitted_params)

    # Plot the original data points
    plt.scatter(x_csv_subset, y_csv_subset, label='Data Subset')

    # Plot the fitted curve
    plt.plot(fitX, fitY, label='Fitted Curve')

    # Set labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitted Double Exponential Curve (Subset)')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()

