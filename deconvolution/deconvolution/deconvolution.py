import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time as time_module
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy import interpolate
from numba import njit, prange


"""
Functions that you might be looking for:

- Deconvolve_SingleExp : deconvolve a constant single-exponential instrument response function
- Deconvolve_DblExp : deconvolve a constant double-exponential instrument response function
- Deconvolve_DblExp_VariableIRF: determine the variability in the instrument response function (double-exponential) 
  from prescribed parts of the dataset and deconvolve

Hannah Verhaal and Demetrios Pagonis, Weber State University, 2024
MIT License 

"""

def Deconvolve_SingleExp(wX, wY, Tau, NIter = 0):
    """
    Deconvolves signal, 'wY', using double exponential kernel at each point

    Parameters:
    wX (np.ndarray): Array of time values
    wY (np.ndarray): Array of signal values
    Tau (float): Time constant for the exponential response function

    Optional parameters:
    NIter (int): Number of iterations for the deconvolution process. default iterates until solution stabilizes

    Returns:
    np.ndarray: Deconvolved signal array

    """
    ForceIterations = 1 if NIter != 0 else 0
    NIter = 100 if NIter == 0 else NIter
    
    time_max = int(10*Tau) # Calculate the desired duration
    N = np.argmin(np.abs(wX - time_max))

    # make X data for kernel
    wX_kernel = wX[:N] - wX[0]
    
    # Calculate delta_x, the spacing between the points in wX_kernel_upsampled
    delta_x = wX_kernel[1] - wX_kernel[0]
    kernel = np.zeros_like(wX_kernel)
    kernel = np.exp(-wX_kernel / Tau) / Tau
    kernel /= np.sum(kernel)* delta_x # Normalize kernel

    wError = np.zeros_like(wY)
    wConv = np.zeros_like(wY)
    wLastConv = np.zeros_like(wY)
    wDest = wY

    LastR2 = 0.01
    R2 = 0.01


    for ii in range(NIter):
        wLastConv[:] = wConv[:]
        
        #do the convolution
        full_conv = np.convolve(wDest, kernel, mode='full')* delta_x

        # Correct the shift for 'full' output by selecting the appropriate portion of the convolution
        wConv = full_conv[:len(wY)]
        
        # Determine error between convoluted signal and original data
        wError[:] = wConv - wY
        
        # Update correlation coefficient
        LastR2 = R2
        R2 = np.corrcoef(wConv, wY)[0, 1] ** 2
        
        # Check for stopping criteria
        if ((abs(R2 - LastR2) / LastR2) * 100 > 0.1) or (ForceIterations == 1):
            wDest[:] = wDest - wError
        else:
            print(f"Stopped deconv at N={ii}, %R2 change={(abs(R2 - LastR2) / LastR2) * 100:.3f}")
            break
    
    return wDest

def Deconvolve_DblExp(wX, wY, Tau1, A1, Tau2, A2, NIter = 0):
    """
    Deconvolves signal, 'wY', using double exponential kernel at each point

    Parameters:
    wX (np.ndarray): Array of time values
    wY (np.ndarray): Array of signal values
    Tau1 (float): Time constant for the first exponential component
    A1 (float): Amplitude for the first exponential component
    Tau2 (float): Time constant for the second exponential component
    A2 (float): Amplitude for the second exponential component
    
    Optional parameters:
    NIter (int): Number of iterations for the deconvolution process. default iterates until solution stabilizes

    Returns:
    np.ndarray: Deconvolved signal array

    """
    ForceIterations = 1 if NIter != 0 else 0
    NIter = 100 if NIter == 0 else NIter
    
    time_max = int(10 * max(Tau1, Tau2)) # Calculate the desired duration
    N = np.argmin(np.abs(wX - time_max))

    # make X data for kernel
    wX_kernel = wX[:N] - wX[0]
    
    
    # create normalized kernel
    delta_x = wX_kernel[1] - wX_kernel[0]
    kernel = np.zeros_like(wX_kernel)
    kernel = (A1 / Tau1) * np.exp(-wX_kernel / Tau1) + (A2 / Tau2) * np.exp(-wX_kernel / Tau2)
    kernel /= np.sum(kernel)* delta_x # Normalize kernel

    wError = np.zeros_like(wY)
    wConv = np.zeros_like(wY)
    wLastConv = np.zeros_like(wY)
    wDest = wY

    LastR2 = 0.01
    R2 = 0.01


    for ii in range(NIter):
        wLastConv[:] = wConv[:]
        
        #do the convolution
        full_conv = np.convolve(wDest, kernel, mode='full')* delta_x

        # Correct the shift for 'full' output by selecting the appropriate portion of the convolution
        wConv = full_conv[:len(wY)]
        
        # Determine error between convoluted signal and original data
        wError[:] = wConv - wY
        
        # Update correlation coefficient
        LastR2 = R2
        R2 = np.corrcoef(wConv, wY)[0, 1] ** 2
        
        # Check for stopping criteria
        if ((abs(R2 - LastR2) / LastR2) * 100 > 0.1) or (ForceIterations == 1):
            wDest = wDest - wError
        else:
            print(f"Stopped deconv at N={ii}, %R2 change={(abs(R2 - LastR2) / LastR2) * 100:.3f}")
            break
    
    return wDest

def Deconvolve_DblExp_VariableIRF(df, directory, base_name, NIter=0, increasing_IRF=False, make_figures=False):
   
    """
    Processes data with variable Instrument Response Functions (IRFs) to determine IRFs and deconvolve them
    
    Parameters:
    df (pd.DataFrame): DataFrame containing all original data. Columns: 
        'time' : datetime objects, evenly spaced
        'signal' : signal to be deconvolved
        'IRF_key' : 1/0 flag where 1 indicates the time periods to be used for fitting IRF
        Optional:
            'IRF_data' : for cases where IRF is fitted to a different time series than 'signal' (e.g. isotopically labeled calibrant)
            'bg_key' : 1/0 flag indicating periods where 
    directory (str): Directory path where output files will be saved
    basename (str): name to use for writing output data. existing data will be overwritten
    
    Optional Parameters:
    NIter (int): Number of iterations for the deconvolution process. default iterates until solution stabilizes
    increasing_IRF (bool): Flag to determine if specific integration intervals are used, Default False
    make_figures (bool): Flag to make figures or not

    Returns:
    np.ndarray: Array of deconvolved signal aligned with time series

    Outputs:
    csv containing deconvolved data
    csv containing fit parameters and timestamps for IRF: {base_name}_IRF.csv
    summary figure pngs (optional)

    """
    start_time=time_module.time()

    if 'IRF_data' not in df.columns:
        df['IRF_data']=df['signal']

    # Drop rows where there are NaN values
    data = df.dropna(subset=['signal', 'IRF_data'])

    # Convert time values to Unix timestamps
    wX = [pd.Timestamp(dt64).timestamp() for dt64 in data['time'].values] 
    wY = data['signal'].values

    # Fit the IRF before deconvolution
    df_IRF = FitIRFs_DblExp(df, directory, base_name, increasing_IRF=increasing_IRF, make_figures=make_figures) 

    wDest = HV_Deconvolve(df, df_IRF, directory, base_name, NIter)

    if make_figures:
        HV_PlotFigures(wX, wY, wDest, directory, base_name)

    # Calculate the integrals
    integral_wY = trapezoid(wY,wX)
    integral_wDest = trapezoid(wDest,wX)
    print("Area ratio, deconvolution/original: {:.4f}".format(integral_wDest/integral_wY))

    # Calculate the total runtime
    end_time = time_module.time()
    total_runtime = end_time - start_time
    print("Total runtime: {:.1f} seconds".format(total_runtime))

    # Return deconvolved data
    return wDest


def DP_DblExp_NormalizedIRF_dec(x, A1, tau1, tau2):
    return A1 * np.exp(-x / tau1) + (1 - A1) * np.exp(-x / tau2)

def DP_DblExp_NormalizedIRF_inc(x, A1, tau1, tau2):
    return 1 - A1 * np.exp(-x / tau1) - (1 - A1) * np.exp(-x / tau2)

def DP_FitDblExp(wY, wX, PtA=None, PtB=None, x0=None, x1=None, y0=None, y1=None, A1=None, tau1=None, tau2=None, increasing_IRF=False):
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
    increasing_IRF (bool, optional): flag for direction of step change
    
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

    if increasing_IRF:
        fit_func = DP_DblExp_NormalizedIRF_inc
    else:
        fit_func = DP_DblExp_NormalizedIRF_dec
    
    popt, pcov = curve_fit(fit_func, wX[PtA:PtB+1] - x0, wY_norm, p0=p0, bounds=([0,0,0],[1,3600,3600]))

    # Generate the fitted curve
    fitX = wX[PtA:PtB+1]
    fitY = fit_func(fitX - x0, *popt) * (NormFactor - y0) + y0
    
    return popt, pcov, fitX, fitY

def FitIRFs_DblExp(df, directory, base_name, increasing_IRF, make_figures): 
    """
    Fits instrument response functions (IRFs) to time-resolved segments of signal based on calibration flags

    Parameters:
    df (pd.DataFrame): Input dataset containing time series and IRF keys
    directory (str): Directory path where output files will be saved
    base_name (str): base name for output files
    increasing_IRF (bool): Flag to determine fitting logic
        - If TRUE, fits IRF to a step-function increase in concentration
        - If FALSE (default behavior), fits IRF to a step-function decrease in concentration

    Returns:
    IRF dataframe
    
    """

    # Extract necessary data from the dataframe
    x_values_datetime = df['time'].values 
    x_values_numeric = np.array([(date - np.datetime64('1970-01-01T00:00:00')).astype('timedelta64[s]').astype(float) for date in x_values_datetime])
    y_values = df['IRF_data'].values
    IRF_key = df['IRF_key'].values

    #track down all the periods where we need to fit an IRF
    intervals = []

    starts = np.where((IRF_key[:-1] == 0) & (IRF_key[1:] == 1))[0] + 1
    ends = np.where((IRF_key[:-1] == 1) & (IRF_key[1:] == 0))[0] + 1

    if IRF_key[0] == 1:
        starts = np.insert(starts, 0, 0)
    if IRF_key[-1] == 1:
        ends = np.append(ends, len(IRF_key))

    for start, end in zip(starts, ends):
        intervals.append((start, end))   

    # set up the big IRF figure
    if make_figures:
        num_columns = 2
        num_rows = int(np.ceil(len(intervals)/2))

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 2*num_rows), squeeze=False) 

    fit_info_list = [['time', 'A1', 'Tau1', 'A2', 'Tau2']]

    for i, (start_index, end_index) in enumerate(zip(starts, ends)):
        
        
        x_subset_numeric = x_values_numeric[start_index:end_index]
        y_subset = y_values[start_index:end_index]
        fitted_params, _, _, fitY = DP_FitDblExp(y_subset, x_subset_numeric, increasing_IRF) 
        
        if make_figures:
            ax = axes[i // num_columns, i % num_columns]
            ax.scatter(x_values_datetime[start_index:end_index], y_subset, label='Signal', color='blue')
            ax.plot(x_values_datetime[start_index:end_index], fitY, label='Fitted IRF', color='black')

            ax.set_xlabel('Time')
            ax.set_ylabel('Signal')
            ax.legend()

            fit_info = f"A1: {fitted_params[0]:.4f}\nTau1: {fitted_params[1]:.4f}\nA2: {1-fitted_params[0]:.4f}\nTau2: {fitted_params[2]:.4f}"
            ax.text(0.3, 0.5, fit_info, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='gray'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # Always add fit info to list for all segments
        if start_index < end_index:  # Ensure there's data in the segment
            fit_info_list.append([x_values_numeric[start_index], fitted_params[0],fitted_params[1],1-fitted_params[0],fitted_params[2]])
   

    if make_figures:
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f'{base_name}_IRF.png'))
        plt.close(fig)

    # Save the fit information as a CSV file
    fit_info_df = pd.DataFrame(fit_info_list)
    fit_info_df.to_csv(os.path.join(directory, f'{base_name}_IRF.csv'), index=False)    

    return pd.read_csv(os.path.join(directory, f'{base_name}_IRF.csv'),header=1)



def HV_Deconvolve(df_data, df_IRF, directory, base_name, NIter): 
    """Performs iterative deconvolution on signal using provided IRF

    Parameters:
    wX (np.ndarray): Array of time values corresponding to wY
    wY (np.ndarray): Array of signal values to be deconvovled
    df_IRF (pd.DataFrame): IRF fit result dataframe
    directory (str): Base directory path where output files will be stored
    basename (str): base name for file outputs
    NIter (int): Number of iterations to perform in deconvolution process
    
    Returns:
    np.ndarray: Array containing deconvolved signal

    """    
    # Convert time values to Unix timestamps
    wX = [pd.Timestamp(dt64).timestamp() for dt64 in df_data['time'].values] 
    wY = df_data['signal'].values

    ForceIterations = 1 if NIter != 0 else 0
    NIter = 100 if NIter == 0 else NIter

    wError = np.zeros_like(wY)
    wConv = np.zeros_like(wY)
    wLastConv = np.zeros_like(wY)
    wDest = wY

    LastR2 = 0.01
    R2 = 0.01

    for ii in range(NIter):
        wLastConv[:] = wConv[:]

        # Do the convolution
        wConv = HV_Convolve(wX, wY, df_IRF)

        wError[:] = wConv - wY
        LastR2 = R2
        R2 = np.corrcoef(wConv, wY)[0, 1] ** 2
        
        if ((abs(R2 - LastR2) / LastR2) * 100 > 1) or (ForceIterations == 1):
            wDest = wDest - wError
        else:
            print(f"Stopped deconv at N={ii}, %R2 change={(abs(R2 - LastR2) / LastR2) * 100:.3f}")
            break

    # Save to CSV
    if 'bg_key' in df_data.columns:
        wDest_bgsub, bg = HV_BG_subtract_data(wX, wDest, df_data['bg_key'].values)
        output_df = pd.DataFrame({'time': wX, 'deconvolution': wDest, 'deconvolution_bgsub' : wDest_bgsub, 'background_signal' : bg})
    else:
        output_df = pd.DataFrame({'time': wX, 'deconvolution': wDest})

    output_filename = os.path.join(directory,f'{base_name}_outputdata.csv')
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
        wKernel /= np.sum(wKernel)/spacing
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
    return wConv

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
    chunk_size = 1000

    # Process signal in chunks
    for start in range(0, len(wX), chunk_size):
        end = min(start + chunk_size, len(wX))  # Ensure the last chunk doesn't exceed the length of wX
        wConv = HV_Convolve_chunk(wX, wY, A1, A2, Tau1, Tau2, wConv, start, end)

    return wConv

def HV_PlotFigures(wX, wY, wDest, directory, base_name):
       
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
    fig_save_path = os.path.join(directory, f"{base_name}_SummaryFigure.png")
    plt.savefig(fig_save_path)
    plt.close()


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
    - Returning background-subtracted signal and interpolated background values
    """
    # Check if inputs are pandas Series and convert to lists if necessary
    if isinstance(wX, pd.Series):
        wX = wX.values.tolist()
    if isinstance(wY, pd.Series):
        wY = wY.values.tolist()
    if isinstance(background_key, pd.Series):
        background_key = background_key.values.tolist()

    background_averages, background_average_times = HV_average_background(wX, wY, background_key)
    
    # Subtract interpolated background
    background_values_interpolated = np.interp(wX, background_average_times, background_averages)
    wY_subtracted_bg = wY - background_values_interpolated

    return wY_subtracted_bg, background_values_interpolated

def HV_average_background( wX, processed_wY, processed_background_key):
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