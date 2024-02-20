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
from numba import njit, prange, objmode
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

def FitIRF(data, csv_filename, directory, time_col, IRF_col, calibration_col): #2024-02-20 optional param appears here
    # Derive base_filename from csv_filename
    base_filename = csv_filename.rstrip('.csv')

    # Extract the x, y, and z values from the data
    x_values_datetime = data['time_col_datetime'].values  # Use the correct column name from the function parameter
    x_values_numeric = np.array([(date - np.datetime64('1970-01-01T00:00:00')).astype('timedelta64[s]').astype(float) for date in x_values_datetime])
    y_values = data[IRF_col].values
    z_values = data[calibration_col].values

    #2024-02-20 somewhere around here there will be if/else logic for defining the times to be used for fitting

    # Identify start and end indices of segments where 'zero flag' == 1
    starts = np.where((z_values[:-1] == 0) & (z_values[1:] == 1))[0] + 1
    ends = np.where((z_values[:-1] == 1) & (z_values[1:] == 0))[0] + 1

    # Ensure covering segment from start if it begins with 1
    if z_values[0] == 1:
        starts = np.insert(starts, 0, 0)
    # Ensure covering segment till end if it ends with 1
    if z_values[-1] == 1:
        ends = np.append(ends, len(z_values))

    # Adjust for only displaying the first 5 IRFs
    display_limit = 5
    num_columns = 2
    num_rows = min(len(starts), display_limit)
    num_rows = (num_rows + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 6), squeeze=False)  # Ensure axes is always 2D

    fit_info_list = [['time', 'A1', 'Tau1', 'A2', 'Tau2']]

    for i, (start_index, end_index) in enumerate(zip(starts, ends)):
        if i < display_limit:  # Limit plotting to first 5
            ax = axes[i // num_columns, i % num_columns]
            x_subset_numeric = x_values_numeric[start_index:end_index]
            y_subset = y_values[start_index:end_index]
            fitted_params, covariance, fitX, fitY = DP_FitDblExp(y_subset, x_subset_numeric)  # Assuming DP_FitDblExp is defined elsewhere

            # DEBUGGING PRINTS
            # print(f"Segment {i+1}: A1={fitted_params[0]}, Tau1={fitted_params[1]}, A2={1-fitted_params[0]}, Tau2={fitted_params[2]}")

            ax.scatter(x_values_datetime[start_index:end_index], y_subset, label='Signal', color='blue')
            ax.plot(x_values_datetime[start_index:end_index], fitY, label='Fitted IRF', color='black')

            ax.set_xlabel('Time')
            ax.set_ylabel('Signal (ncps)')
            ax.set_title(f'Cal {i + 1}')
            ax.legend()

            fit_info = f"A1: {fitted_params[0]:.4f}\nTau1: {fitted_params[1]:.4f}\nA2: {1-fitted_params[0]:.4f}\nTau2: {fitted_params[2]:.4f}"
            ax.text(0.3, 0.5, fit_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='gray'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # Always add fit info to list for all segments
        if start_index < end_index:  # Ensure there's data in the segment
            fit_info_list.append([x_values_numeric[start_index], fitted_params[0],fitted_params[1],1-fitted_params[0],fitted_params[2]])

    plt.tight_layout()
    #plt.show()

    # Save the figure as a PNG file in the desired directory
    plt.savefig(os.path.join(directory, 'Figures', f'{base_filename}_InstrumentResponseFunction.png'))

    # Save the fit information as a CSV file
    fit_info_df = pd.DataFrame(fit_info_list)

    fit_info_df.to_csv(os.path.join(directory, 'Output Data', f'{base_filename}_InstrumentResponseFunction.csv'), index=False, header=False)    

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
    output_data_dir = directory + '/Output Data/'
    
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
        #wConv = wConv/points_per_interval

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

    # Save to CSV
    output_df = pd.DataFrame({'time': wX, 'Deconvolved Data': wDest})

    output_filename = f"{output_data_dir}{date_str}_Deconvolved_Data.csv"
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

def HV_subtract_background(wY, wDest, wX, background_averages, background_average_times):
    # Interpolate the background averages over the entire dataset
    background_values_interpolated = np.interp(wX, background_average_times, background_averages)

    # Subtract the background from the original data
    wY_subtracted_bg = wY - background_values_interpolated

    # Subtract the background from the deconvolved data
    wDest_subtracted_bg = wDest - background_values_interpolated

    return wY_subtracted_bg, wDest_subtracted_bg, background_values_interpolated

def HV_PlotFigures(wX, wY, wDest, directory):
    print("Starting HV_PlotFigures...")
    
    # Directory to save figures
    save_dir = directory + "/Figures/"
    print(f"Saving figures to: {save_dir}")
    
    # Convert timestamps back to datetime
    times = pd.to_datetime(wX, unit='s')

    # DEGUG to confirm first few data points
    print(f"First 5 times: {times[:5]}")
    print(f"First 5 original data points: {wY[:5]}")
    print(f"First 5 deconvolved data points: {wDest[:5]}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, wY, label='Original Data')
    plt.plot(times, wDest, label='Deconvolved Data')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Original and Deconvolved Signal')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure before calling plt.show()
    fig_save_path = os.path.join(save_dir, "Original_and_Deconvolved_Signal.png")
    plt.savefig(fig_save_path)
    print(f"Figure saved to: {fig_save_path}")
    
    # Display the plot
    plt.show()

def HV_ProcessFlights(directory, datafile, NIter, SmoothError, time_col, IRF_col, calibration_col, data_col, data): #2024-02-20 there will be a new optional parameter here for looking after the flagged period for fitting IRF (default False)
    start_time=time_module.time()

    base_str = datafile.rstrip('.csv')

    # Create output directory path
    output_directory = os.path.join(directory, 'Output Data')
    IRF_filename = os.path.join(output_directory, f'{base_str}_InstrumentResponseFunction.csv')

    # Drop rows where there are NaN values
    data = data.dropna(subset=[data_col, IRF_col])

    wX = [pd.Timestamp(dt64).timestamp() for dt64 in data['time_col_datetime'].values]
    wY = data[data_col].values
    # Background = data[background_col].values

    # Fit the IRF before deconvolution
    FitIRF(data, datafile, directory, time_col, IRF_col, calibration_col) #2024-02-20 optional parameter gets passed through to FitIRF
    
    IRF_data = pd.read_csv(IRF_filename)

    # Deconvolution for CSV data
    wDest = np.zeros_like(wY)
    wDest = HV_Deconvolve(wX, wY, wDest, IRF_data, SmoothError, NIter, datafile, directory)

    # try:
    HV_PlotFigures(wX, wY, wDest, directory)
    print("HV_PlotFigures called successfully.")
# except Exception as e:
    #     print(f"Error calling HV_PlotFigures: {e}")

    # Calculate the integrals
    integral_wY = trapz(wY,wX)
    integral_wDest = trapz(wDest,wX)

    print("Area ratio: {:.4f}".format(1+(integral_wDest-integral_wY)/integral_wY))

    # Calculate the total runtime
    end_time = time_module.time()
    total_runtime = end_time - start_time
    print("Total runtime: {:.1f} seconds".format(total_runtime))

    print('hello')
   
