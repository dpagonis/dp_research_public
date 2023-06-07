import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.signal import resample
import pandas as pd
from datetime import datetime, timedelta
import glob
import os
import time
from scipy.integrate import trapz
from scipy import interpolate
from scipy import signal


def AdjGuess(wG, wE, NSmooth):
    if NSmooth == 0 or NSmooth == 1:
        wG = wG.astype(float)  # Convert wG to float data type
        wG -= wE  # subtract errors
    elif NSmooth > 1:
        wE_Smooth = np.convolve(wE, np.ones(NSmooth) / NSmooth, mode='same')
        wG = wG.astype(float)  # Convert wG to float data type
        wG -= wE_Smooth  # subtract error
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

def Deconvolve_DblExp(wX, wY, wDest, Tau1, A1, Tau2, A2, NIter, SmoothError):
    # Deconvolution algorithm for DOUBLE EXPONENTIAL instrument function
    # Takes input data wX (time) and wY (signal), writes deconvolved output to wDest
    # Double-exponential instrument function defined by Tau1, A1, Tau2, A2
    # NIter: number of iterations of the deconvolution algorithm, 0 is autostop
    # SmoothError: number of time points to use for error smoothing. Set to 0 for no smoothing
    
    # Delete existing iteration_ii.png debugging files
    existing_files = glob.glob("debugplots/iteration_*.png")
    for file_path in existing_files:
        os.remove(file_path)
    
    ForceIterations = 1 if NIter != 0 else 0
    NIter = 100 if NIter == 0 else NIter
    
    N = int(10 * max(Tau1, Tau2)) # Calculate the desired duration

    # make X data for kernel
    wX_kernel = wX[:N] - wX[0]
    
    # Calculate the desired number of points per one-second interval
    points_per_interval = 10

    # Create an array of indices for the original and upsampled data
    old_indices = np.arange(len(wX))
    new_indices = np.linspace(0, len(wX) - 1, len(wY) * points_per_interval)
    old_indices_kernel = np.arange(len(wX_kernel))
    new_indices_kernel = np.linspace(0, len(wX_kernel) - 1, len(wX_kernel) * points_per_interval)

    # Upsample
    wY_upsampled = np.interp(new_indices, old_indices, wY)
    wX_kernel_upsampled = np.interp(new_indices_kernel, old_indices_kernel, wX_kernel)
    
    # Calculate delta_x, the spacing between the points in wX_kernel_upsampled
    delta_x = wX_kernel_upsampled[1] - wX_kernel_upsampled[0]

    kernel = np.zeros_like(wX_kernel_upsampled)
    wError = np.zeros_like(wY_upsampled)
    wConv = np.zeros_like(wY_upsampled)
    wLastConv = np.zeros_like(wY_upsampled)
    wDest_upsampled = wY_upsampled

    LastR2 = 0.01
    R2 = 0.01


    for ii in range(NIter):
        wLastConv[:] = wConv[:]
        
        # define the kernel (instrument response function) and do the convolution
        kernel = (A1 / Tau1) * np.exp(-wX_kernel_upsampled / Tau1) + (A2 / Tau2) * np.exp(-wX_kernel_upsampled / Tau2)
        full_conv = np.convolve(wDest_upsampled, kernel, mode='full') * delta_x

        # Correct the shift for 'full' output by selecting the appropriate portion of the convolution
        wConv = full_conv[:len(wY_upsampled)]
        
        wError[:] = wConv - wY_upsampled
        LastR2 = R2
        R2 = np.corrcoef(wConv, wY_upsampled)[0, 1] ** 2
        
        if ((abs(R2 - LastR2) / LastR2) * 100 > 1) or (ForceIterations == 1):
            wDest_upsampled = AdjGuess(wDest_upsampled, wError, SmoothError)
        else:
            print(f"Stopped deconv at N={ii}, %R2 change={(abs(R2 - LastR2) / LastR2) * 100:.3f}")
            break

        # Make and save figure showing progress for debugging
        fig, axs = plt.subplots()
        axs.plot(wY_upsampled, color='blue', label='Data')
        axs.plot(wError, color='red', label='Error')
        axs.plot(wDest_upsampled, color='green', label='Deconvolved')
        axs.plot(wConv, color='purple', label='Reconstructed Data')
        axs.legend()
        plt.savefig(f'debugplots/iteration_{ii}.png')  # save the figure to file
        plt.close()  # close the figure to free up memory
    
    #downsample 
    wDest = resample(wDest_upsampled, len(wX))
    return wDest

def HV_kernel(time, A1, Tau1, A2, Tau2):
    """Computes the kernel function based on given parameters."""
    resolution = 0.1  # Fixed resolution of 0.1 units
    max_tau = max(Tau1, Tau2)
    duration = 10 * max_tau
    num_points = int(duration / resolution)

    wX_kernel_upsampled = np.linspace(0, duration, num_points)
    kernel = (A1 / Tau1) * np.exp(-wX_kernel_upsampled / Tau1) + (A2 / Tau2) * np.exp(-wX_kernel_upsampled / Tau2)
    
    # Set values outside the kernel range to the values of the first and last data points
    kernel[:num_points] = kernel[0]
    kernel[num_points:] = kernel[-1]
    
    return kernel

def HV_Convolve(WDest, wX, FitResults):
    """Perform convolution at each point and store results in WConv."""

    # Read the fit parameters from the CSV file
    FitResults = pd.read_csv('C:/Users/hjver/Documents/dp_research_public/deconvolution/data/2019_08_07_InstrumentResponseFunction.csv')

    # Create interpolation functions for the parameters
    A1_func = interpolate.interp1d(FitResults['time'], FitResults['A1'])
    Tau1_func = interpolate.interp1d(FitResults['time'], FitResults['Tau1'])
    A2_func = interpolate.interp1d(FitResults['time'], FitResults['A2'])
    Tau2_func = interpolate.interp1d(FitResults['time'], FitResults['Tau2'])

    # Interpolate the parameters for the times in wX
    A1 = A1_func(wX)
    Tau1 = Tau1_func(wX)
    A2 = A2_func(wX)
    Tau2 = Tau2_func(wX)

    # Ensure that WDest is a NumPy array
    WDest = np.array(WDest)

    # Loop through each point in wX
    for i in range(len(wX)):

        # Compute the kernel for the specific point
        kernel = HV_kernel(wX[i], A1[i], Tau1[i], A2[i], Tau2[i])

        # Perform the dot product between the kernel and data points
        wConv = np.dot(wY, kernel)

        # Store the result in WDest
        WDest[i] = wConv

    return WDest

def HV_Deconvolve(wX, wY, wDest):
    
    # Delete existing iteration_ii.png debugging files
    existing_files = glob.glob("debugplots/iteration_*.png")
    for file_path in existing_files:
        os.remove(file_path)
    
    ForceIterations = 1 if NIter != 0 else 0
    NIter = 100 if NIter == 0 else NIter
    
    N = int(10 * max(Tau1, Tau2)) # Calculate the desired duration

    # make X data for kernel
    wX_kernel = wX[:N] - wX[0]
    
    # Calculate the desired number of points per one-second interval
    points_per_interval = 10

    # Create an array of indices for the original and upsampled data
    old_indices = np.arange(len(wX))
    new_indices = np.linspace(0, len(wX) - 1, len(wY) * points_per_interval)
    old_indices_kernel = np.arange(len(wX_kernel))
    new_indices_kernel = np.linspace(0, len(wX_kernel) - 1, len(wX_kernel) * points_per_interval)

    # Upsample
    wY_upsampled = np.interp(new_indices, old_indices, wY)
    wX_kernel_upsampled = np.interp(new_indices_kernel, old_indices_kernel, wX_kernel)
    
    # Calculate delta_x, the spacing between the points in wX_kernel_upsampled
    delta_x = wX_kernel_upsampled[1] - wX_kernel_upsampled[0]

    kernel = np.zeros_like(wX_kernel_upsampled)
    wError = np.zeros_like(wY_upsampled)
    wConv = np.zeros_like(wY_upsampled)
    wLastConv = np.zeros_like(wY_upsampled)
    wDest_upsampled = wY_upsampled

    LastR2 = 0.01
    R2 = 0.01


    for ii in range(NIter):
        wLastConv[:] = wConv[:]
        
        # Do the convolution
        full_conv = HV_Convolve(wDest, wX, kernel)

        # Correct the shift for 'full' output by selecting the appropriate portion of the convolution
        wConv = full_conv[:len(wY_upsampled)]
        
        wError[:] = wConv - wY_upsampled
        LastR2 = R2
        R2 = np.corrcoef(wConv, wY_upsampled)[0, 1] ** 2
        
        if ((abs(R2 - LastR2) / LastR2) * 100 > 1) or (ForceIterations == 1):
            wDest_upsampled = AdjGuess(wDest_upsampled, wError, SmoothError)
        else:
            print(f"Stopped deconv at N={ii}, %R2 change={(abs(R2 - LastR2) / LastR2) * 100:.3f}")
            break

        # Make and save figure showing progress for debugging
        fig, axs = plt.subplots()
        axs.plot(wY_upsampled, color='blue', label='Data')
        axs.plot(wError, color='red', label='Error')
        axs.plot(wDest_upsampled, color='green', label='Deconvolved')
        axs.plot(wConv, color='purple', label='Reconstructed Data')
        axs.legend()
        plt.savefig(f'debugplots/iteration_{ii}.png')  # save the figure to file
        plt.close()  # close the figure to free up memory
    
    #downsample 
    wDest = resample(wDest_upsampled, len(wX))
    return wDest

if __name__ == "__main__":
    start_time = time.time()
    
    # Load the data from the CSV file
    directory = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/'
    # directory = 'C:/Users/demetriospagonis/Box/github/dp_research_public/deconvolution/data/'
    datafile = '2019_08_07_InstrumentResponseFunction.csv'

    data = pd.read_csv(directory+datafile)
    wX = data['time'].values
    wY = data['original_signal'].values

    # Set the parameters for deconvolution
    # Tau1 = 4.9327  # Replace with the desired value
    # A1 = 0.7072  # Replace with the desired value
    # Tau2 = 55.2182  # Replace with the desired value
    # A2 = 1.0-A1  # Replace with the desired value
    # NIter = 0  # Replace with the desired value
    SmoothError = 0  # Replace with the desired value
    
    # Deconvolution
    wDest = np.zeros_like(wY)
    wDest = HV_Deconvolve(wX, wY, wDest)

    end_time = time.time()

    # Plot the figures
    plt.figure(figsize=(10, 6))

    # Original time series and deconvolution
    plt.subplot(2, 1, 1)
    plt.plot(wX, wY, label='Original')
    plt.plot(wX, wDest, label='Deconvolution')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()

    # Deconvolution only
    plt.subplot(2, 1, 2)
    plt.plot(wX, wDest, label='Deconvolution',color='C1')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.tight_layout()

    # Calculate the integrals
    integral_wY = trapz(wY,wX)
    integral_wDest = trapz(wDest,wX)
    
    # # Print the integrals
    # print("Integral of wY: {:.1f}".format(integral_wY))
    # print("Integral of wDest: {:.1f}".format(integral_wDest))

    print("Area ratio: {:.4f}".format(1+(integral_wDest-integral_wY)/integral_wY))
    # Calculate the total runtime
    total_runtime = end_time - start_time
    print("Total runtime: {:.1f} seconds".format(total_runtime))

    
    # Save the figure as a PNG file
    base_str = datafile.rstrip('.csv')
    plt.savefig(directory + f'{base_str}_Deconvolution.png')
    plt.show()

    # Save wX, wY, and wDest to a CSV file
    output_data = pd.DataFrame({'time': wX, 'original_signal': wY, 'deconvolved_signal': wDest})
    output_file = directory + f'{base_str}_Deconvolution.csv'
    output_data.to_csv(output_file, index=False)
