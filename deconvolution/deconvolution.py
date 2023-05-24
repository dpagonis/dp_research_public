import numpy as np
import matplotlib.pyplot as plt

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





if __name__ == "__main__":
    # this is where you put code for testing these functions
    # Sample inputs for testing
    wG = np.array([1, 2, 3, 4, 5])
    wE = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    NSmooth = 1
    # Call the function and store the modified wG
    modified_wG = AdjGuess(wG, wE, NSmooth)
    # Plot the original and modified wG
    plt.plot(wG, label='Original wG')
    plt.plot(modified_wG, label='Modified wG')
    plt.legend()
    plt.show()
