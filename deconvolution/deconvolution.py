import numpy as np

def DP_AdjGuess(wG, wE, NSmooth):
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

# Sample inputs for testing
wG = np.array([1, 2, 3, 4, 5])
wE = np.array([0.5, 0.5, 0.1, 0.5, 0.5])
NSmooth = 1

# Call the function and store the modified wG
modified_wG = DP_AdjGuess(wG, wE, NSmooth)

# Print the modified wG array
print(modified_wG)
