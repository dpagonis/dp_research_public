import numpy as np

def AdjGuess(wG, wE, NSmooth):
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


