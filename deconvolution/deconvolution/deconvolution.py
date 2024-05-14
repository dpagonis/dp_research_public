import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.signal import resample
import pandas as pd
import time as time_module
import glob
import os
from scipy.integrate import trapz
from scipy import interpolate
from numba import njit, prange

def AdjGuess(wG, wE, NSmooth):
    """
    Adjusts initial guess array, 'wG', by subtracting error array, 'wE'
    
    Parameters:
    wG (np.ndarray): Initial guess array to be adjusted
    wE (np.ndarray): Error array to be subtracted from guess
    NSmooth (int): Optional smoothing parameter
                - If 0 or 1: No smoothing applied, direct subtraction of error
                - If > 1: Applies a moving average smoothing to the error before subtraction
    
    Returns:
    np.ndarray: Adjusted guess array after error subtraction

    Description:
    Directly modifies guess array 'wG' based on error array 'wE' by the following:
    - Checking the 'NSmooth' value to determine the smoothing behavior.
    - If 'NSmooth' is 0 or 1, the function subtracts the error array 'wE' directly from 'wG'.
    - If 'NSmooth' is greater than 1, it first applies a moving average filter to 'wE' to smooth the error values. Smoothed error is then subtracted from 'wG'.
    - Subtraction adjusts 'wG' by reducing discrepancies between'wG', and original data
    - Output is the adjusted 'wG', which represents a guess closer to the true data pattern after accounting for errors
    """
    if NSmooth == 0 or NSmooth == 1:
        wG = wG.astype(float)  # Convert wG to float data type
        wG -= wE  # subtract errors
    elif NSmooth > 1:
        wE_Smooth = np.convolve(wE, np.ones(NSmooth) / NSmooth, mode='same')
        wG = wG.astype(float)  # Convert wG to float data type
        wG -= wE_Smooth  # subtract error
    else:
        raise ValueError("DP_AdjGuess in DP_Deconvolution was passed a bad value for NSmooth")
    
    # Optional: Ensure no negative values in the guess
    #wG = np.where(wG < 0, 0, wG)
    return wG

def DP_DblExp_NormalizedIRF(x, A1, tau1, tau2):
    """
    Determines double exponential decay function, normalized by amplitudes
    
    Parameters:
    x (np.ndarray or float): Input array representing time
    A1 (float): Amplitude of the first exponential component, between 0 and 1
    tau1 (float): Time constant for the first exponential decay
    tau2 (float): Time constant for the second exponential decay
    
    Returns:
    np.ndarray or float: Calculated values of double exponential function for each input in 'x'

    Description:
    Models signal decay using two exponential components combined into a single function by the following:
    - Multiplying the amplitude 'A1' by an exponential decay function of 'x' over 'tau1'
    - Adding result to the product of (1 - A1) and another exponential decay function of 'x' over 'tau2'
    """
    return A1 * np.exp(-x / tau1) + (1 - A1) * np.exp(-x / tau2)

def DP_FitDblExp(wY, wX, PtA=None, PtB=None, x0=None, x1=None, y0=None, y1=None, A1=None, tau1=None, tau2=None):
    """
    Fits a double exponential decay function to signal, 'wY', against time, 'wX'
    
    Parameters:
    wY (np.ndarray): Array containing original signal
    wX (np.ndarray): Array of time values corresponding to 'wY'
    PtA (int, optional): Starting index for the fitting range. Defaults to start of 'wX'
    PtB (int, optional): Ending index for the fitting range. Defaults to end of 'wX'
    x0 (float, optional): Starting time value for the fitting range. Overrides 'PtA' if provided
    x1 (float, optional): Ending time value for the fitting range. Overrides 'PtB' if provided
    y0 (float, optional): Baseline value for normalization. Defaults to the mean of the last 20 points of `wY`
    y1 (float, optional): Normalization factor. Defaults to the value of `wY` at `PtA`
    A1 (float, optional): Initial guess for the amplitude of the first exponential component. Defaults to 0.5
    tau1 (float, optional): Initial guess for the time constant of the first exponential component. Defaults to 1
    tau2 (float, optional): Initial guess for the time constant of the second exponential component. Defaults to 80
    
    Returns:
    A tuple containing:
        - popt (np.ndarray): Optimal values for parameters A1, tau1, tau2
        - pcov (np.ndarray): Covariance matrix of the fitted parameters
        - fitX (np.ndarray): Time values used for the fit
        - fitY (np.ndarray): Fitted double exponential function values
    
    Description:
    Applies nonlinear least squares to fit a double exponential model to data in a specified range.
    The model combines two exponential decay functions by the following:
    - Selects segment of signal specified by `PtA` and `PtB` or `x0` and `x1`
    - Normalizes segment based on `y0` and `y1`
    - Fits normalized signal to a double exponential function using the initial guesses for amplitudes and time constants
    - Fitting is constrained within specified bounds to ensure realistic parameters
    - Recalculates the fitted curve over the original time values to produce output that can be compared to original signal
    """
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

def FitIRF(data, csv_filename, directory, time_col, IRF_col, calibration_col, FIREXint=False): 
    """
    Fits instrument response functions (IRFs) to time-resolved segments of signal based on calibration flags

    Parameters:
    data (pd.DataFrame): Input dataset containing time series and IRF data
    csv_filename (str): Name of input CSV file
    directory (str): Directory path where output files will be saved
    time_col (str): Column name for time values in dataset
    IRF_col (str): Column name for IRF values in dataset
    calibration_col (str): Column name for calibration flags in dataset
    FIREXint (bool, optional): Flag to determine fitting logic
                                - If TRUE, identifies transitions from 1 to 0
                                - Defaults to False

    Returns:
    None: Function saves fitted IRF plots and parameters to the specified directory
    
    Description:
    IRFs are fit by the following steps:
    - Identify fitting intervals based on calibration flags and the FIREXint flag
    - Fit a double exponential decay model to data within each interval
    - Plot and save fitted models with original data for comparison
    - Fit parameters saved in CSV file and plots saved as PNG files
    """
    # Derive base_filename from csv_filename
    base_filename = csv_filename.rstrip('.csv')

    # Extract the x, y, and z values from the data
    x_values_datetime = data[time_col].values  # Use the correct column name from the function parameter
    x_values_numeric = np.array([(date - np.datetime64('1970-01-01T00:00:00')).astype('timedelta64[s]').astype(float) for date in x_values_datetime])
    y_values = data[IRF_col].values
    z_values = data[calibration_col].values

    intervals = []

    # if/else logic for defining the times used for fitting
    if FIREXint:
        # Identify transition from 1 to 0 and fit 100 points after transition
        transitions = np.where((z_values[:-1] == 1) & (z_values[1:] == 0))[0] + 1
        starts = transitions
        # Calculate ends directly, vectorized manner
        ends = np.minimum(transitions + 100, len(z_values))
        intervals = list(zip(transitions, ends))
    else:
        # Fit where 'zero flag' == 1
        starts = np.where((z_values[:-1] == 0) & (z_values[1:] == 1))[0] + 1
        ends = np.where((z_values[:-1] == 1) & (z_values[1:] == 0))[0] + 1

        if z_values[0] == 1:
            starts = np.insert(starts, 0, 0)
        if z_values[-1] == 1:
            ends = np.append(ends, len(z_values))

        for start, end in zip(starts, ends):
            intervals.append((start, end))   

    # Adjust for only displaying the first few IRFs
    display_limit = 10
    num_columns = 2
    num_rows = min(len(starts), display_limit)
    num_rows = (num_rows + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 10), squeeze=False)  # Ensure axes is always 2D

    fit_info_list = [['time', 'A1', 'Tau1', 'A2', 'Tau2']]

    for i, (start_index, end_index) in enumerate(zip(starts, ends)):
        
        
        x_subset_numeric = x_values_numeric[start_index:end_index]
        y_subset = y_values[start_index:end_index]
        fitted_params, covariance, fitX, fitY = DP_FitDblExp(y_subset, x_subset_numeric) 
        
        if i < display_limit:  # Limit plotting to first 5 fits
            ax = axes[i // num_columns, i % num_columns]
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
   
    # Plot IRF
    plt.tight_layout()

    # Save the figure as a PNG file in the desired directory
    plt.savefig(os.path.join(directory, 'Figures', f'{base_filename}_InstrumentResponseFunction.png'))
    plt.close(fig)

    # Save the fit information as a CSV file
    fit_info_df = pd.DataFrame(fit_info_list)

    fit_info_df.to_csv(os.path.join(directory, 'Output Data', f'{base_filename}_InstrumentResponseFunction.csv'), index=False, header=False)    

def Deconvolve_DblExp(wX, wY, wDest, Tau1, A1, Tau2, A2, NIter, SmoothError, points_per_interval = 0):
    """
    Deconvolves signal, 'wY', using double exponential kernel at each point

    Parameters:
    wX (np.ndarray): Array of time values
    wY (np.ndarray): Array of signal values
    wDest (np.ndarray): Array to store deconvolved signal
    Tau1 (float): Time constant for the first exponential component
    A1 (float): Amplitude for the first exponential component
    Tau2 (float): Time constant for the second exponential component
    A2 (float): Amplitude for the second exponential component
    NIter (int): Number of iterations for the deconvolution process
    SmoothError (int): Smoothing factor applied to error in each iteration
    points_per_interval (int, optional): Number of points per interval for upsampling. Defaults to 0

    Returns:
    np.ndarray: Deconvolved signal array

    Description:
    Iteratively deconvolves signal based on IRF's through the following steps:
    - Upsampling original signal
    - Defining IRF using amplitudes and decay constants, then convolving kernel with the upsampled signal
    - Comparing convolved signal with upsampled original to compute errors, which are then smoothed if specified
    - Adjusting deconvolved signal estimate iteratively based on the smoothed errors, refining the estimate to minimize errors
    - Iterations continue until the change in the correlation coefficient between consecutive iterations is less than 0.1%
    - Deconvolved signal is downsampled back to original resolution
    """
    # Delete existing iteration_ii.png debugging files
    existing_files = glob.glob("debugplots/iteration_*.png")
    for file_path in existing_files:
         os.remove(file_path)
    
    ForceIterations = 1 if NIter != 0 else 0
    NIter = 100 if NIter == 0 else NIter
    
    time_max = int(10 * max(Tau1, Tau2)) # Calculate the desired duration
    N = np.argmin(np.abs(wX - time_max))

    # make X data for kernel
    wX_kernel = wX[:N] - wX[0]
    
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
        kernel /= np.sum(kernel)* delta_x # Normalize kernel
        full_conv = np.convolve(wDest_upsampled, kernel, mode='full')* delta_x

        # Correct the shift for 'full' output by selecting the appropriate portion of the convolution
        wConv = full_conv[:len(wY_upsampled)]
        
        # Determine error between convoluted signal and original data
        wError[:] = wConv - wY_upsampled
        
        # Update correlation coefficient
        LastR2 = R2
        R2 = np.corrcoef(wConv, wY_upsampled)[0, 1] ** 2
        
        # Check for stopping criteria
        if ((abs(R2 - LastR2) / LastR2) * 100 > 0.1) or (ForceIterations == 1):
            wDest_upsampled = AdjGuess(wDest_upsampled, wError, SmoothError*points_per_interval)
        else:
            print(f"Stopped deconv at N={ii}, %R2 change={(abs(R2 - LastR2) / LastR2) * 100:.3f}")
            break
    
    #downsample 
    wDest = resample(wDest_upsampled, len(wX))
    
    return wDest

def HV_Deconvolve(wX, wY, wDest, IRF_data, SmoothError, NIter, datafile, directory, debug_plots=False): 
    """Performs iterative deconvolution on signal using provided IRF

    Parameters:
    wX (np.ndarray): Array of time values corresponding to wY
    wY (np.ndarray): Array of signal values to be deconvovled
    wDest (np.ndarray): Array to store deconvolved signal
    IRF_data (np.ndarray): Array of IRF values
    SmoothError (int): Smoothing factor applied to error in each iteration
    NIter (int): Number of iterations to perform in deconvolution process
    datafile (str): Path to file used for output file naming
    directory (str): Base directory path where output files will be stored
    debug_plots (bool, optional): Flag to control creation of debugging plots, Default False
    
    Returns:
    np.ndarray: Array containing deconvovled signal

    Description:
    Deconvolves signal by the following steps:
    - Upsampling original signal 
    - Convolving the upsampled signal with the IRF
    - Calculating the error between convoluted signal and upsampled original
    - Iteratively adjusting deconvolved signal guess based on smoothed error
    - Repeating process for a specified number of iterations, or until changes in the correlation coefficient between iterations fall below 0.1%
    - Downsampling deconvolved signal to original resolution
    """    
    # Path for saving CSV file
    output_data_dir = directory + '/Output Data/'
    
    # Delete existing iteration_ii.png debugging files if debug_plots is True
    if debug_plots:
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

        # Make and save figure showing progress for debugging if debug_plots is True
        if debug_plots:
            fig, axs = plt.subplots() 
            axs.plot(wY_upsampled, color='blue', label='Data')
            axs.plot(wError, color='red', label='Error')
            axs.plot(wDest_upsampled, color='green', label='Deconvolved')
            axs.plot(wConv, color='purple', label='Reconstructed Data')
            axs.legend()
            fig.savefig(f'debugplots/iteration_{ii}.png')
            plt.close(fig)  
            
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
    """
    Runs in paralell to perform convolution over separate chunks of data

    Parameters:
    wX (np.ndarray): Array of time values 
    wY (np.ndarray): Array of signal values
    A1 (np.ndarray): Array of amplitudes for the first exponential decay component at each time point
    A2 (np.ndarray): Array of amplitudes for the second exponential decay component at each time point
    Tau1 (np.ndarray): Array of time constants for the first exponential decay component at each time point
    Tau2 (np.ndarray): Array of time constants for the second exponential decay component at each time point
    wConv (np.ndarray): Output array where convolved signal is stored
    start (int): Starting index of the chunk of data to process
    end (int): Ending index of the chunk of data to process

    Description:
    Determines convolution of signal 'wY' with a kernel for each time point by the following:
    - Extracting A1, A2, Tau1, Tau2 values for each point in the chunk
    - Creating convolution kernel
    - Determining length of the kernel, handling boundary conditions by padding with the first value of 'wY' if needed
    - Flipping kernel for convolution
    - Extracting relevant segment of 'wY' for convolution
    - Performing convolution by computing the dot product of the kernel with corresponding signal
    - Storing convolution result in wConv output array
    """
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
    """
    Performs convolution of signal with IRF for each segment
    
    Parameters:
    wX (np.ndarray): Array of time values 
    wY (np.ndarray): Array of signal values
    IRF_Data (pd.DataFrame): DataFrame containing IRF parameters, columns for 'time', 'A1', 'Tau1', 'Tau2'
    
    Returns:
    np.ndarray: Array containing convolved signal, represents output that would be observed by
    an instrument with specified IRF

    Description:
    Convolves signal 'wY' with an IRF by the following:
    - Creating interpolation functions for IRF parameters
    - Interpolating IRF parameters for time in 'wX' to obtain arrays
    - Setting a chunk size for processing data
    - Iterating over 'wY' in chunks and performing a convolution for each chunk with 'HV_Convolve_chunk'
    - Returning convovled signal in 'wConv' array 
    """

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

    # Set chunk size
    chunk_size = 1000

    # Process signal in chunks
    for start in range(0, len(wX), chunk_size):
        end = min(start + chunk_size, len(wX))  # Ensure the last chunk doesn't exceed the length of wX
        HV_Convolve_chunk(wX, wY, A1, A2, Tau1, Tau2, wConv, start, end)

    return wConv

def HV_BG_subtract_data(wX, wY, background_key):
    """
    Subtracts interpolated background from signal

    Parameters:
    wX (pd.Series or list): Array or list of time values
    wY (pd.Series or list): Array or list of signal values
    background_key (pd.Series or list): Array or list of background flags

    Returns: 
    tuple containing:
    - np.ndarray: Array of signal values with background subtracted
    - np.ndarray: Arrate of interpolated background values

    Description:
    Interpolated background is subtracted from signal 'wY' by the following:
    - Checking inputs and converting to lists if necessary
    - Interpolating background values with 'HV_interpolate_background' to get background averages, times
    - Subtracting interpolated background from 'wY' with 'HV_subtract_background'
    - Returning background-subtracted signal and interpolated background values
    """
    # Check if inputs are pandas Series and convert to lists if necessary
    if isinstance(wX, pd.Series):
        wX = wX.values.tolist()
    if isinstance(wY, pd.Series):
        wY = wY.values.tolist()
    if isinstance(background_key, pd.Series):
        background_key = background_key.values.tolist()

    background_averages, background_average_times = HV_interpolate_background(wX, wY, background_key)
    
    # Subtract interpolated background
    wY_subtracted_bg, background_values_interpolated = HV_subtract_background(wX, wY, background_averages, background_average_times)

    return wY_subtracted_bg, background_values_interpolated

def HV_interpolate_background( wX, processed_wY, processed_background_key):
    """
    Interpolates background values from signal

    Parameters:
    wX (np.ndarray): Array of time values
    processed_wY (np.ndarray): Array of pre-processed signal values, pre-processing background signal is up to the user
    processed_background_key (np.ndarray): Array containing background flags, pre-processing background flags is up to the user
    
    Returns:
    tuple containing:
    - list: List of background averages for each segment
    - list: List of average time points corresponding to each background segment
    
    Description:
    Interpolates background from 'processed_wY' by the following:
    - Calculating start and end indices of each background measurement segment based on 'processed_background_key'
    - Calculating average signal value and corresponding time for each background segment
    - Returning lists of background averages and average time points
    """
    # Calculate average for each segment, store averages with their time points
    background_averages = []
    background_average_times = []

    # Find the start and end indices of each background measurement
    bg_start_indices = np.where(np.diff(processed_background_key) == 1)[0] + 1
    bg_end_indices = np.where(np.diff(processed_background_key) == -1)[0]

    # Handle the case where the Background starts with 1
    if processed_background_key[0] == 1:
        bg_start_indices = np.insert(bg_start_indices, 0, 0)
    # Handle the case where the Background ends with 1
    if processed_background_key[-1] == 1:
        bg_end_indices = np.append(bg_end_indices, len(processed_background_key) - 1)

    if len(bg_start_indices) > len(bg_end_indices):
        # Remove the unmatched start indices
        bg_start_indices = bg_start_indices[:len(bg_end_indices)]
    elif len(bg_end_indices) > len(bg_start_indices):
        # Remove the unmatched end indices
        bg_end_indices = bg_end_indices[:len(bg_start_indices)]

    # Verify that there are equal numbers of start and end indices
    assert len(bg_start_indices) == len(bg_end_indices), "Number of start and end indices for background measurements do not match"

    for start, end in zip(bg_start_indices, bg_end_indices):
        # Calculate average for each background segment and its corresponding time
        segment_average = np.mean(processed_wY[start:end+1])
        segment_time = np.mean(wX[start:end+1])
        
        # Using average of start and end times as the representative time point for each segment
        background_average_times.append(segment_time)
        background_averages.append(segment_average)
        
    return background_averages, background_average_times

def HV_subtract_background(wX, wY, background_averages, background_average_times):
    """
    Subtracts interpolated background values from signal

    Parameters:
    wX (np.ndarray): Array of time values
    wY (np.ndarray): Array of signal values
    background_averages (list): List of average background signal values for each segment
    background_average_times (list): List of average time points corresponding to each background segment

    Returns:
    tuple containing:
    - np.ndarray: Array of signal values with background subtracted
    - np.ndarray: Array of interpolated background values

    Description:
    Subtracts background from 'wY' by the following:
    - Interpolating background averages over entire dataset
    - Subtracting interpolated background values from 'wY'
    - Returning background-subtracted signal and interpolated background values
    """
    # Interpolate the background averages over the entire dataset
    background_values_interpolated = np.interp(wX, background_average_times, background_averages)

    # Subtract interpolated background from signal
    wY_subtracted_bg = wY - background_values_interpolated

    return wY_subtracted_bg, background_values_interpolated

def HV_PlotFigures(wX, wY, wDest, directory):
    """
    Plots original, deconvolved signals vs. time, saves plot to specified directory

    Parameters:
    wX (np.ndarray): Array of Unix timestamps 
    wY (np.ndarray): Array of original signal values
    wDest (np.ndarray): Array of deconvolved signal values
    directory (str): Base directory where plot will be saved as PNG

    Description:
    Generates a plot of original and deconvovled signals by the following:
    - Creating directory to save figures
    - Converting Unix timestamps in 'wX' to datetime format
    - Plotting original signal 'wY' and deconvolved signal 'wDest'against time
    - Saving plot as PNG file in specified directory
    """
    # Directory to save figures
    save_dir = directory + "/Figures/"
    
    # Convert timestamps back to datetime
    times = pd.to_datetime(wX, unit='s')
    
    # Plot original, deconvolved signal vs. time
    plt.figure(figsize=(10, 6))
    plt.plot(times, wY, label='Original Data')
    plt.plot(times, wDest, label='Deconvolved Data')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Original and Deconvolved Signal')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    fig_save_path = os.path.join(save_dir, "Original_and_Deconvolved_Signal.png")
    plt.savefig(fig_save_path)
    plt.close()

def HV_ProcessFlights(directory, datafile, NIter, SmoothError, time_col, IRF_col, calibration_col, data_col, data, background_col=None, FIREXint=False):
   
    """
    Processes flight data by fitting Instrument Response Functions (IRFs) and performing deconvolution
    
    Parameters:
    directory (str): Directory path where output files will be saved
    datafile (str): Name of CSV file containing data
    NIter (int): Number of iterations for the deconvolution process
    SmoothError (int): Smoothing factor applied to error in each iteration
    time_col (str): Column in dataset that contains time data
    IRF_col (str): Column in dataset containing IRF fitting data
    calibration_col (str): Column containing calibration flags
    data_col(str): Column containing signal
    data (pd.DataFrame): DataFrame containing all original data
    background_col (str, optional): Column containing background flags
    FIREXint (bool, optional): Flag to determine if specific integration intervals are used, Default False

    Returns:
    np.ndarray: Array of deconvolved signal aligned with time series

    Description:
    Processes flight data by the following:
    - Creating output directory path, determining IRF filename
    - Dropping rows in DataFrame with NaN values
    - Converting time values in 'wX' to timestamps
    - Fitting IRF before deconvolution with 'FitIRF' function
    - Performing deconvolution of signal with 'HV_Deconvolve' function
    - Plotting original, deconvolved signals versus time with 'HV_PlotFigures' function
    - Calculating area ratio of original and deconvolved signal
    - Returning deconvovled signal in 'wDest' array
    """
    start_time=time_module.time()

    base_str = datafile.rstrip('.csv')

    # Create output directory path
    output_directory = os.path.join(directory, 'Output Data')
    IRF_filename = os.path.join(output_directory, f'{base_str}_InstrumentResponseFunction.csv')

    # Drop rows where there are NaN values
    data = data.dropna(subset=[data_col, IRF_col])

    # Convert time values to Unix timestamps
    wX = [pd.Timestamp(dt64).timestamp() for dt64 in data[time_col].values] 
    wY = data[data_col].values

    # Fit the IRF before deconvolution
    FitIRF(data, datafile, directory, time_col, IRF_col, calibration_col, FIREXint=FIREXint) 
    
    # Load IRF data
    IRF_data = pd.read_csv(IRF_filename)

    # Deconvolution for CSV data
    wDest = np.zeros_like(wY)
    wDest = HV_Deconvolve(wX, wY, wDest, IRF_data, SmoothError, NIter, datafile, directory)

    # Plot Signal versus time
    HV_PlotFigures(wX, wY, wDest, directory)

    # Calculate the integrals
    integral_wY = trapz(wY,wX)
    integral_wDest = trapz(wDest,wX)
    print("Area ratio: {:.4f}".format(1+(integral_wDest-integral_wY)/integral_wY))

    # Calculate the total runtime
    end_time = time_module.time()
    total_runtime = end_time - start_time
    print("Total runtime: {:.1f} seconds".format(total_runtime))

    # Return deconvolved data
    return wDest
