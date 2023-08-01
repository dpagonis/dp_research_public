import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.signal import resample
import pandas as pd
from datetime import datetime, timedelta, time
import time as time_module
import glob
import os
from scipy.integrate import trapz
from scipy import interpolate
from numba import njit, prange
from scipy.stats import linregress


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
    base_datetime = datetime(1904, 1, 1)  # Change to your base datetime

    # Convert numpy.int64 to regular integer
    igor_timestamp = int(igor_timestamp)

    # Calculate timedelta
    timedelta_seconds = timedelta(seconds=igor_timestamp)

    return base_datetime + timedelta_seconds

def ict_to_datetime(ict_timestamp, measurement_date):
    base_datetime = datetime.strptime(measurement_date, "%Y%m%d")
    return base_datetime + timedelta(seconds=int(ict_timestamp))

def FitIRF(csv_filename, directory):
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

        # Code for datasets w/o pre-normalized data
        # Get the corresponding subset of column values from the data
        # column_subset = data.loc[start_index:end_index - 1, ['I_127_Hz', 'IH20_145_Hz']]
        # Modify the data points
        # normalized_y = (y_subset / (column_subset['I_127_Hz'] + column_subset['IH20_145_Hz'])) * 10 ** 6

    # Call the fitting function with the modified subset of data
        fitted_params, covariance, fitX, fitY = DP_FitDblExp(y_subset, x_subset_numeric)

        # Create the subplot for the current subset
        if num_rows > 1:
            ax = axes[i // num_columns, i % num_columns]
        else:
            ax = axes[i % num_columns]

        # Plot the original data points
        ax.scatter(x_values_datetime[start_index:end_index], y_subset, label='m/z 191: 15N HNO3I', color='blue')
        ax.plot(x_values_datetime[start_index:end_index], fitY, label='Fitted IRF', color='black', zorder=2)

        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal (ncps)')
        ax.set_title(f'Cal {i + 1}')

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

    # Define the save directory for figures
    save_dir = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/Figures/'

    # Define the directory for CSV files
    csv_dir = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/Output Data'

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as a PNG file in the desired directory
    plt.savefig(save_dir + f'{date_str}_InstrumentResponseFunction.png')
    plt.close()

    # Save the fit information as a CSV file with the extracted date in the CSV directory
    filename = f'{date_str}_InstrumentResponseFunction.csv'
    fit_info_df = pd.DataFrame(fit_info_list)
    fit_info_df.to_csv(os.path.join(csv_dir, filename), index=False, header=False)

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

def HV_Deconvolve(wX, wY, wDest, IRF_data, SmoothError, NIter, datafile, directory):

    # Path for saving CSV file
    output_data_dir = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/Output Data/'
    
    # Delete existing iteration_ii.png debugging files
    existing_files = glob.glob("debugplots/iteration_*.png")
    for file_path in existing_files:
        os.remove(file_path)
    
    ForceIterations = 1 if NIter != 0 else 0
    NIter = 100 if NIter == 0 else NIter

    # Calculate the desired number of points per one-second interval
    points_per_interval = 10

    # Create an array of indices for the original and upsampled data
    old_indices = np.arange(len(wX))
    new_indices = np.linspace(0, len(wX) - 1, len(wY) * points_per_interval)

    # Upsample
    wY_upsampled = np.interp(new_indices, old_indices, wY)
    wX_upsampled = np.interp(new_indices, old_indices, wX)

    wError = np.zeros_like(wY_upsampled)
    wConv = np.zeros_like(wY_upsampled)
    wLastConv = np.zeros_like(wY_upsampled)
    wDest_upsampled = wY_upsampled

    LastR2 = 0.01
    R2 = 0.01

    for ii in range(NIter):
        wLastConv[:] = wConv[:]
        
        # Do the convolution
        wConv = HV_Convolve(wX_upsampled, wY_upsampled, IRF_data)
        wConv = wConv/points_per_interval

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

    # Extracting date from input CSV filename
    date_str = datafile[:10]

    # Define the output directory for deconvolved data
    output_data_dir = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/Output Data/'

    # Save to CSV
    output_df = pd.DataFrame({'time': wX, 'HNO3_190_Hz': wDest})

    output_filename = f"{output_data_dir}{date_str}_Deconvolved_HNO3Data.csv"
    output_df.to_csv(output_filename, index=False)

    return wDest

@njit(parallel=True)
def HV_Convolve_chunk(wX, wY, A1, A2, Tau1, Tau2, wConv, start, end):
    for idx in prange(start, end):
        # Get A and tau values at time i
        A1_i = A1[idx]
        A2_i = A2[idx]
        Tau1_i = Tau1[idx]
        Tau2_i = Tau2[idx]

        # Create the kernel
        max_tau = max(Tau1_i, Tau2_i)
        spacing = wX[1] - wX[0]  # assuming wX is evenly spaced
        num_steps = int(10 * max_tau / spacing)
        wX_kernel = np.linspace(0, 10 * max_tau, num_steps)
        wKernel = (A1_i / Tau1_i) * np.exp(-wX_kernel / Tau1_i) + (A2_i / Tau2_i) * np.exp(-wX_kernel / Tau2_i)
        wKernel = np.flip(wKernel)

        # Pad wY_i manually if necessary
        if idx < num_steps:
            # Use wY[0] for padding
            padding = np.full(num_steps - idx-1, wY[0])
            wY_i = np.concatenate((padding, wY[:idx+1]))
        else:
            wY_i = wY[idx-num_steps+1 : idx+1]


        # Perform the convolution
        wConv[idx] = np.dot(wY_i, wKernel)


def HV_Convolve(wX, wY, IRF_Data):
    """Perform convolution at each point and store results in WConv."""
    
    # Create interpolation functions for the parameters
    A1_func = interpolate.interp1d(IRF_Data['time'], IRF_Data['A1'], fill_value=(IRF_Data['A1'].values[0], IRF_Data['A1'].values[-1]), bounds_error=False)
    Tau1_func = interpolate.interp1d(IRF_Data['time'], IRF_Data['Tau1'], fill_value=(IRF_Data['Tau1'].values[0], IRF_Data['Tau1'].values[-1]), bounds_error=False)
    A2_func = interpolate.interp1d(IRF_Data['time'], IRF_Data['A2'], fill_value=(IRF_Data['A2'].values[0], IRF_Data['A2'].values[-1]), bounds_error=False)
    Tau2_func = interpolate.interp1d(IRF_Data['time'], IRF_Data['Tau2'], fill_value=(IRF_Data['Tau2'].values[0], IRF_Data['Tau2'].values[-1]), bounds_error=False)

    # Interpolate the parameters for the times in wX
    A1 = A1_func(wX)
    Tau1 = Tau1_func(wX)
    A2 = A2_func(wX)
    Tau2 = Tau2_func(wX)

    # Prepare destination array
    wConv = np.zeros_like(wY)

    # Set your chunk size
    chunk_size = 1000

    # Process the data in chunks
    for start in range(0, len(wX), chunk_size):
        end = min(start + chunk_size, len(wX))  # Ensure the last chunk doesn't exceed the length of wX
        HV_Convolve_chunk(wX, wY, A1, A2, Tau1, Tau2, wConv, start, end)

    return wConv

def HV_interpolate_background(Background, wY, wX):
    # Find the start and end indices of each background measurement
    bg_start_indices = np.where(np.diff(Background) == 1)[0] + 1
    bg_end_indices = np.where(np.diff(Background) == -1)[0]

    # Handle the case where the Background starts with 1
    if Background[0] == 1:
        bg_start_indices = np.insert(bg_start_indices, 0, 0)

    # Handle the case where the Background ends with 1
    if Background[-1] == 1:
        bg_end_indices = np.append(bg_end_indices, len(Background) - 1)

    if len(bg_start_indices) > len(bg_end_indices):
        # Remove the unmatched start indices
        bg_start_indices = bg_start_indices[:len(bg_end_indices)]
    elif len(bg_end_indices) > len(bg_start_indices):
        # Remove the unmatched end indices
        bg_end_indices = bg_end_indices[:len(bg_start_indices)]

    # Verify that there are equal numbers of start and end indices
    assert len(bg_start_indices) == len(bg_end_indices), "Number of start and end indices for background measurements do not match"

    # Calculate a separate average for each segment and store these averages along with their time points
    background_averages = []
    background_average_times = []

    for start, end in zip(bg_start_indices, bg_end_indices):
        # Exclude the first 10s and last 5s from each segment
        start += 10
        end -= 5
        # Make sure start is still before end after adjusting indices
        if start >= end:
            print(f"Skipping background segment from {start} to {end} because it's too short after excluding the first 10s and last 5s")
            continue
        segment_average = np.mean(wY[start:end+1])
        background_averages.append(segment_average)
        # assuming that each segment's representative time point is the average of its start and end times
        segment_time = np.mean(wX[start:end+1])
        background_average_times.append(segment_time)
        
    return background_averages, background_average_times

def HV_get_common_time_and_interpolated_data(wX, wY, wDest, wX_ict, wY_ict, wY_subtracted_bg, wDest_subtracted_bg, background_values_interpolated, date_str_ict):
    # Convert both time series to datetime
    wX_datetime = [igor_to_datetime(ts) for ts in wX]
    wX_ict_datetime = [ict_to_datetime(ts, date_str_ict) for ts in wX_ict]

    # Convert datetime objects to timestamps
    wX_timestamp = [dt.timestamp() for dt in wX_datetime]
    wX_ict_timestamp = [dt.timestamp() for dt in wX_ict_datetime]

    # Make sure that both time series start at the same point and have the same length
    common_start = max(wX_datetime[0], wX_ict_datetime[0])
    common_end = min(wX_datetime[-1], wX_ict_datetime[-1])
    common_length = min(len(wX_datetime), len(wX_ict_datetime))
    common_wX = np.linspace(common_start.timestamp(), common_end.timestamp(), common_length)
    
    # Interpolate all series to the common time basis
    interp_wY = np.interp(common_wX, wX_timestamp, wY)
    interp_wY_ict = np.interp(common_wX, wX_ict_timestamp, wY_ict)
    interp_wDest = np.interp(common_wX, wX_timestamp, wDest)
    interp_background_values = np.interp(common_wX, wX_timestamp, background_values_interpolated)
    interp_wY_subtracted_bg = np.interp(common_wX, wX_timestamp, wY_subtracted_bg )
    interp_wDest_subtracted_bg = np.interp(common_wX, wX_timestamp, wDest_subtracted_bg)

    return common_wX, interp_wY, interp_wY_ict, interp_wDest, interp_background_values, interp_wY_subtracted_bg, interp_wDest_subtracted_bg

def HV_subtract_background(wY, wDest, wX, background_averages, background_average_times):
    # Interpolate the background averages over the entire dataset
    background_values_interpolated = np.interp(wX, background_average_times, background_averages)

    # Subtract the background from the original data
    wY_subtracted_bg = wY - background_values_interpolated

    # Subtract the background from the deconvolved data
    wDest_subtracted_bg = wDest - background_values_interpolated

    return wY_subtracted_bg, wDest_subtracted_bg, background_values_interpolated

def HV_generate_figures(common_wX_datetime, interp_wY, interp_wY_ict, interp_wDest, interp_wY_subtracted_bg, interp_wDest_subtracted_bg, interp_background_values, directory, date_str):
    
    # Directory to save figures
    save_dir = directory + "Figures/"

    # Original, Deconvolved Data & ICARTT Data Time Series
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(common_wX_datetime, interp_wY, label='HNO3')
    plt.plot(common_wX_datetime, interp_wY_ict, label=' CO')
    plt.plot(common_wX_datetime, interp_wDest, label='Deconvolved HNO3')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Original and Deconvolved Signal')
    plt.legend()
    # plt.savefig(save_dir + f"{date_str}_Original_and_Deconvolved_Signal.png")
    plt.subplot(2, 1, 2)
    plt.plot(common_wX_datetime, interp_wDest, label='Deconvolved HNO3', color='C1')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Deconvolved Signal Only')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(save_dir + f"{date_str}_Deconvolved_Signal_Only.png")
    plt.close()

    # Configure x-axis tick formatter as datetime
    time_formatter = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')

    # Apply the formatter to the x-axis
    plt.gca().xaxis.set_major_formatter(time_formatter)

    # Original Data with BG Subtracted
    plt.figure()
    plt.plot(common_wX_datetime, interp_wY_subtracted_bg, label='Original HNO3 - BG')
    plt.plot(common_wX_datetime, interp_wY_ict, color='red', label='CO')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Original HNO3 Data with Background Subtraction')
    plt.legend()
    # plt.savefig(save_dir + f"{date_str}_Original_HNO3_with_Background_Subtraction.png")
    plt.close()

    # Original Data & Interpolated BG
    plt.figure()
    plt.plot(common_wX_datetime, interp_wY, label='Original HNO3')
    plt.plot(common_wX_datetime, interp_background_values, label='Interpolated BG')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Original HNO3 Data and Interpolated Background')
    plt.legend()
    # plt.savefig(save_dir + f"{date_str}_Original_HNO3_Data_and_Interpolated_Background.png")
    plt.close()

    # Deconvolved Data with BG Subtracted
    plt.figure()
    plt.plot(common_wX_datetime, interp_wDest_subtracted_bg, label='Deconvolved HNO3 - BG')
    plt.plot(common_wX_datetime, interp_wY_ict, color='red', label='CO')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Deconvolved HNO3 Data with Background Subtraction')
    plt.legend()
    # plt.savefig(save_dir + f"{date_str}_Deconvolved_HNO3_Data_with_Background_Subtraction.png")
    plt.close()

    # Deconvolved Data & Interpolated BG
    plt.figure()
    plt.plot(common_wX_datetime, interp_wDest, label='Deconvolved HNO3')
    plt.plot(common_wX_datetime, interp_background_values, label='Interpolated BG')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Deconvolved HNO3 Data and Interpolated Background')
    plt.legend()
    # plt.savefig(save_dir + f"{date_str}_Deconvolved_HNO3_Data_and_Interpolated_Background.png")
    plt.close()

    # Convert all items to datetime, if not already
    for i, item in enumerate(common_wX_datetime):
        if not isinstance(item, datetime):
            # Assuming they're timestamps if not datetime objects
            common_wX_datetime[i] = datetime.fromtimestamp(item)


    # Define the time interval 1:00am - 2:30am
    start_time = time(1, 0)  # 1:00am
    end_time = time(2, 30)  # 2:30am

    # Filter based on the time
    time_mask = [(t.time() >= start_time) and (t.time() <= end_time) for t in common_wX_datetime]

    # Filter data where CO < 300
    CO_mask = np.array(interp_wY_ict) >= 300

    # Combine the masks (i.e., time & CO filter)
    combined_mask = [a and b for a, b in zip(time_mask, CO_mask)]

    # Apply the combined mask
    filtered_wY = np.array(interp_wY)[combined_mask]
    filtered_wY_ict = np.array(interp_wY_ict)[combined_mask]
    filtered_wDest = np.array(interp_wDest)[combined_mask]

    # Original data correlation plot
    plt.figure(figsize=(6, 4))
    slope, intercept, r_value, _, _ = linregress(filtered_wY_ict, filtered_wY)
    plt.scatter(filtered_wY_ict, filtered_wY, marker='.', color='b')
    plt.plot(filtered_wY_ict, intercept + slope*filtered_wY_ict, 'r', label=f'y={slope:.2f}x+{intercept:.2f}, $R^2$={r_value**2:.2f}')
    plt.xlabel('CO')
    plt.ylabel('HNO3')
    plt.title('Original Data Correlation Plot')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig(save_dir + f"{date_str}_Original_Data_Correlation_Plot.png")
    plt.close()

    # Deconvolved data correlation plot
    plt.figure(figsize=(6, 4))
    slope, intercept, r_value, _, _ = linregress(filtered_wY_ict, filtered_wDest)
    plt.scatter(filtered_wY_ict, filtered_wDest, marker='.', color='b')
    plt.plot(filtered_wY_ict, intercept + slope*filtered_wY_ict, 'r', label=f'y={slope:.2f}x+{intercept:.2f}, $R^2$={r_value**2:.2f}')
    plt.xlabel('CO')
    plt.ylabel('Deconvolved HNO3')
    plt.title('Deconvolved Data Correlation Plot')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig(save_dir + f"{date_str}_Deconvolved_Data_Correlation_Plot.png")
    plt.close()


    # Step Function
    plt.figure(figsize=(10, 8))
    plt.step(common_wX_datetime, interp_wDest, where='post', label='Deconvolved HNO3', color='blue')
    plt.step(common_wX_datetime, interp_wY, where='post', label='HNO3', color='green')
    plt.step(common_wX_datetime, interp_wY_ict, where='post', label='CO', color='red')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('2019_08_21 Step Function')
    plt.legend()
    # plt.savefig(save_dir + f"{date_str}_Interpolated_Start_Stop_Data.png")
    plt.close()

def Get_ICT_Filename(csv_filename):
    date_str = csv_filename[:10]
    ict_date = date_str.replace("_", "")
    ict_filename = f"FIREXAQ-DACOM_DC8_{ict_date}_R1.ict"
    return ict_filename

def HV_ProcessFlights(directory, datafile, ict_file, NIter, SmoothError):

    start_time=time_module.time()

    base_str = datafile.rstrip('.csv')
    date_str = datafile[:10]

    # Create output directory path
    output_directory = os.path.join(directory, 'Output Data')
    IRF_filename = os.path.join(output_directory, f'{date_str}_InstrumentResponseFunction.csv')

    data = pd.read_csv(directory+datafile)

    # Drop rows where 'HNO3_190_Hz' or 'HNO3_191_Hz' has NaN values
    data = data.dropna(subset=['HNO3_190_Hz', 'HNO3_191_Hz'])

    wX = data['time'].values
    wY = data['HNO3_190_Hz'].values
    Background = data['ZeroKey'].values

    # Fit the IRF before deconvolution
    FitIRF(datafile, directory)
    
    IRF_data = pd.read_csv(IRF_filename)

    # Deconvolution for CSV data
    wDest = np.zeros_like(wY)
    wDest = HV_Deconvolve(wX, wY, wDest, IRF_data, SmoothError, NIter, datafile, directory)

    # Calculate the integrals
    integral_wY = trapz(wY,wX)
    integral_wDest = trapz(wDest,wX)

    print("Area ratio: {:.4f}".format(1+(integral_wDest-integral_wY)/integral_wY))

    # Parse date string form ICT file
    date_str_ict = ict_file[18:26]

    # Load the ICT file data
    ict_data = pd.read_csv(directory+ict_file, skiprows=36)
    wX_ict = ict_data['Time_Start'].values  # Assuming 'Time_Start' is your time column
    wY_ict = ict_data['CO_DACOM'].values

    # Replace any CO values below 0 with NaN
    wY_ict[wY_ict < 0] = np.nan

    # Calculate the background averages and times
    background_averages, background_average_times = HV_interpolate_background(Background, wY, wX)

    # Subtract BG from original and deconvolved data
    wY_subtracted_bg, wDest_subtracted_bg, background_values_interpolated = HV_subtract_background(wY, wDest, wX, background_averages, background_average_times)

    # Get the common time and interpolated data
    common_wX, interp_wY, interp_wY_ict, interp_wDest, interp_background_values, interp_wY_subtracted_bg, interp_wDest_subtracted_bg  = HV_get_common_time_and_interpolated_data(wX, wY, wDest, wX_ict, wY_ict, wY_subtracted_bg, wDest_subtracted_bg, background_values_interpolated, date_str_ict)
    
    # Convert the common_wX timestamps to datetime objects
    common_wX_datetime = [datetime.fromtimestamp(ts) for ts in common_wX]
    
    # Calculate the total runtime
    end_time = time_module.time()
    total_runtime = end_time - start_time
    print("Total runtime: {:.1f} seconds".format(total_runtime))

    # Call the generate_figures function and pass the required arguments
    HV_generate_figures(common_wX_datetime, interp_wY, interp_wY_ict, interp_wDest, interp_wY_subtracted_bg, interp_wDest_subtracted_bg, interp_background_values, directory, date_str)

if __name__ == "__main__":
    
    # Load data from csv and ict files
    directory = 'C:/Users/hjver/Documents/dp_research_public/deconvolution/data/'
    datafiles = ['2019_08_07_HNO3Data.csv']

    # Assuming iterations and smooth error are the same for all flights, if not you can adjust
    iterations = 5
    smooth_err = 0

    for datafile in datafiles:
        ict_file = Get_ICT_Filename(datafile)
        print(f"Processing {datafile}...")
        HV_ProcessFlights(directory, datafile, ict_file, iterations, smooth_err)
    