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
        y0 = np.mean(wY[-20:])

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

    # Load the data from the CSV file
    data = pd.read_csv('C:/Users/hjver/Documents/dp_research_public/deconvolution/data/2019_08_07_HNO3Data.csv')

    # Extract the x, y, and z values from the data
    x_values = data['time'].values
    y_values = data['HNO3_191_Hz'].values
    z_values = data['CalKey'].values

    # Find the indices where z_values change from 1 to 0
    change_indices = np.where((z_values[:-1] == 1) & (z_values[1:] == 0))[0]

    # Define the number of data points to include after each change
    data_points_after_change = 300

    # Define the size of the subplot grid
    num_subsets = len(change_indices)
    num_columns = 2
    num_rows = (num_subsets + num_columns - 1) // num_columns

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 6))

    # Iterate over the subsets and plot the data
    for i, change_index in enumerate(change_indices):
        # Calculate the start and end indices for each subset
        start_index = change_index
        end_index = start_index + data_points_after_change

        # Get the subset of x and y values
        x_subset = x_values[start_index:end_index]
        y_subset = y_values[start_index:end_index]

        # Call the fitting function with the subset of data
        fitted_params, covariance, fitX, fitY = DP_FitDblExp(y_subset, x_subset, PtA=start_index, PtB=end_index)
        print(fitted_params)

        # # Generate x values for plotting the curve
        # x_plot = np.linspace(min(x_subset), max(x_subset), 100)

        # # Evaluate the fitted double exponential function at x_plot
        # y_plot = DP_DblExp_NormalizedIRF(x_plot, *fitted_params)

        # Create the subplot for the current subset
        if num_rows > 1:
            ax = axes[i // num_columns, i % num_columns]
        else:
            ax = axes[i % num_columns]

        # Plot the original data points
        ax.scatter(x_subset, y_subset, label='Data Subset', color='blue')

        # Plot the fitted curve
        ax.plot(fitX, fitY, label='Fitted Curve', color='red', zorder=2)

        # Set labels and title for the subplot
        ax.set_xlabel('Time')
        ax.set_ylabel('Hz')
        ax.set_title(f'Fitted Double Exponential Curve (Subset {i+1})')

        # Show legend for the subplot
        ax.legend()

        # Display fit information
        fit_info = f"A1: {fitted_params[0]:.4f}\n" \
               f"tau1: {fitted_params[1]:.4f}\n" \
               f"tau2: {fitted_params[2]:.4f}"
        ax.text(0.7, 0.5, fit_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='gray'))

        # Adjust the spacing between subplots
        plt.tight_layout()

    # Display all the subplots in the same window
    plt.show()






