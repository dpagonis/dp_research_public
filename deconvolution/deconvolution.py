import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime, timedelta


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

def igor_to_datetime(igor_timestamp):
    base_datetime = datetime(1904, 1, 1)
    return base_datetime + timedelta(seconds=igor_timestamp)

def plot_and_save_data(csv_filename, directory):
    # Load the data from the CSV file
    data = pd.read_csv(directory+csv_filename)

    # Extract the date from the CSV filename
    date_str = csv_filename[:10]  # Extract the first 10 characters as the date

    # Extract the x, y, and z values from the data
    x_values_numeric = data['time'].values
    x_values_datetime = data['time'].apply(igor_to_datetime).values
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

    # Create a list to store the fit information
    fit_info_list = [['time', 'A1', 'Tau1', 'A2', 'Tau2']]

    # Iterate over the subsets and plot the data
    for i, change_index in enumerate(change_indices):
        # Calculate the start and end indices for each subset
        start_index = change_index
        end_index = start_index + data_points_after_change

        # Get the subset of x and y values
        x_subset_numeric = x_values_numeric[start_index:end_index]
        x_subset_datetime = x_values_datetime[start_index:end_index]
        y_subset = y_values[start_index:end_index]

        # Get the corresponding subset of column values from the data
        column_subset = data.loc[start_index:end_index - 1, ['I_127_Hz', 'IH20_145_Hz']]

        # Modify the data points
        normalized_y = (y_subset / (column_subset['I_127_Hz'] + column_subset['IH20_145_Hz'])) * 10 ** 6

        # Call the fitting function with the modified subset of data
        fitted_params, covariance, fitX, fitY = DP_FitDblExp(normalized_y, x_subset_numeric)

        # Create the subplot for the current subset
        if num_rows > 1:
            ax = axes[i // num_columns, i % num_columns]
        else:
            ax = axes[i % num_columns]

        # Plot the original data points
        ax.scatter(x_values_datetime[start_index:end_index], normalized_y, label='m/z 191: 15N HNO3I', color='blue')
        ax.plot(x_values_datetime[start_index:end_index], fitY, label='Fitted IRF', color='black', zorder=2)

        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal (ncps)')
        ax.set_title(f'Cal {i + 1}')

        # Add the fit information to the plot. currently redundant with the fit info box below
        # num_params = len(fitted_params)
        # if num_params == 3:
        #     fit_info = f'A1={fitted_params[0]:.2f}, T1={fitted_params[1]:.2f}, A2={1-fitted_params[0]:.2f}, T2={fitted_params[2]:.2f}'
        # elif num_params == 4:
        #     fit_info = f'A1={fitted_params[0]:.2f}, T1={fitted_params[1]:.2f}, A2={fitted_params[2]:.2f}, T2={fitted_params[3]:.2f}'
        # else:
        #     fit_info = 'Unknown fit information'
        # ax.text(0.05, 0.9, fit_info, transform=ax.transAxes)

        # Add a legend to the plot
        ax.legend()
        # Display fit information
        fit_info = f"A1: {fitted_params[0]:.4f}\n" \
                   f"tau1: {fitted_params[1]:.4f}\n" \
                   f"A2: {1-fitted_params[0]:.4f}\n" \
                   f"tau2: {fitted_params[2]:.4f}"
        ax.text(0.3, 0.5, fit_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='gray'))
        # Create a DateFormatter for the time only
        time_formatter = mdates.DateFormatter('%H:%M:%S')

        # Apply the formatter to the x-axis
        ax.xaxis.set_major_formatter(time_formatter)

        fit_info_list.append([x_subset_numeric[0], fitted_params[0], fitted_params[1], 1 - fitted_params[0],
                              fitted_params[2]])

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(directory + f'{date_str}_InstrumentResponseFunction.png')

    # Close the figure
    plt.close()

    # Save the fit information as a CSV file with the extracted date
    filename = f'{date_str}_InstrumentResponseFunction.csv'
    fit_info_df = pd.DataFrame(fit_info_list)
    fit_info_df.to_csv(directory + filename, index=False, header=False)

    # Display all the subplots in the same window
    plt.show()

if __name__ == "__main__":
    csv_filename = '2019_08_07_HNO3Data.csv'  # Replace with the actual filename
    directory ='C:/Users/demetriospagonis/Box/github/dp_research_public/deconvolution/data/' #DP
    # directory ='C:/Users/hjver/Documents/dp_research_public/deconvolution/data/' # HV
    plot_and_save_data(csv_filename,directory)
