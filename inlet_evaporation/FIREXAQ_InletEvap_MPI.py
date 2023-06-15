from mpi4py import MPI
import numpy as np
import pandas as pd
from inletevap import *

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get the rank of the current task
size = comm.Get_size() # get the total number of tasks

#Fixed inlet parameters for CU AMS, FIREX-AQ Inlet configuration
TubingInnerDiameter = 0.0047 #m
ResidenceTime = 0.5 #s
ResidenceTimeTAT = 0.019 #s 

# Load data on root process and distribute it among all processes
if rank == 0:
    data = pd.read_csv('FIREX_InletEvap_Input_Test.csv')
    chunks = np.array_split(data, size) # split data into chunks
else:
    chunks = None

chunk = comm.scatter(chunks, root=0) # distribute chunks among processes

# Each process computes true OA for its own chunk of data
AirT = chunk['AirTemp_K']
OA_meas = chunk['OA_ugsm3']
Pres = chunk['Pressure_mbar']
CabinT = chunk['CabinTemp_K']
TAT = chunk['TAT_K']

true_OA = np.zeros_like(chunk)

tolerance = 1e-6 # Define a tolerance level

for i in range(len(chunk)):
    MyInlet = inletevap.Inlet(AirT[i],CabinT[i],TAT[i],Pres[i],TubingInnerDiameter,ResidenceTime,ResidenceTimeTAT)
    VBS = inletevap.VBS_FIREX(AirT[i],Pres[i])
    OA_iter = OA_meas[i]
    
    while True:
        VBS.SetOAConc(OA_iter)
        MyInlet.CalcOAEvap(VBS)
        OA_evap = MyInlet.OAMFR * OA_iter
        
        if abs(OA_evap - OA_meas[i]) < tolerance:  # If the difference is less than the tolerance, break the loop
            break
            
        # If not, adjust OA_iter based on error between OA_evap and OA_meas_std
        OA_iter = OA_iter * (OA_meas[i] / OA_evap)  # This is a simple adjustment, modify as needed.
        
    true_OA[i] = OA_iter


print(f"Task {rank} of {size} complete!")

# Gather all results to root process
results = comm.gather(true_OA, root=0)

# Root process stitches the results back together
if rank == 0:
    final_result = np.concatenate(results)
    # convert the numpy array to pandas dataframe
    df = pd.DataFrame(final_result)
    # write the dataframe to a csv file
    df.to_csv('FIREX_InletEvap_Output.csv', index=False)
