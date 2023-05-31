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
    # Read data from CSV file using pandas
    csv_file_path = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/2019_08_07_HNO3Data.csv'
    data = pd.read_csv(csv_file_path)

    # Define the indices or ranges of the subsets of data
    subsets = [
        (2107, 2407),  # Subset 1
        (5525, 5825),  # Subset 2
        (8998, 9298),   # Subset 3
        (12689, 12989),  # Subset 4
        # Add more subsets as needed
    ]

    num_subsets = len(subsets)
    num_columns = 2  # Number of columns in the subplot grid

    num_rows = (num_subsets + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 8))

    for i, subset in enumerate(subsets):
        # Extract the subset of data to plot and fit
        start_index, end_index = subset
        x_csv_subset = data['time'][start_index:end_index]  # Replace 'x_column' with the actual column name for x values
        y_csv_subset = data['HNO3_191_Hz'][start_index:end_index]  # Replace 'y_column' with the actual column name for y values

        # Call the fitting function with the subset of data
        fitted_params, covariance, fitX, fitY = DP_FitDblExp(y_csv_subset, x_csv_subset)
        print(fitted_params)

        # Generate x values for plotting the curve
        x_plot = np.linspace(min(x_csv_subset), max(x_csv_subset), 100)

        # Evaluate the fitted double exponential function at x_plot
        y_plot = DP_DblExp_NormalizedIRF(x_plot, *fitted_params)

        # Calculate the subplot indices based on the grid
        row_index = i // num_columns
        col_index = i % num_columns

        # Create a new subplot for each subset
        if num_rows > 1:
            ax = axes[row_index, col_index]
        else:
            ax = axes[col_index]

        # Plot the original data points
        ax.scatter(x_csv_subset, y_csv_subset, label='Data Subset')

        # Plot the fitted curve
        ax.plot(fitX, fitY, label='Fitted Curve')

        # Set labels and title for the subplot
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Fitted Double Exponential Curve (Subset {i+1})')

        # Show legend for the subplot
        ax.legend()

    # Hide unused subplots
    if num_subsets < num_rows * num_columns:
        for i in range(num_subsets, num_rows * num_columns):
            if num_rows > 1:
                ax = axes[i // num_columns, i % num_columns]
            else:
                ax = axes[i % num_columns]
            ax.axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display all the subplots
    plt.show()


