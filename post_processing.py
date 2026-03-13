from nses import *
from matplotlib import pyplot as plt
import os
import pandas as pd

result_filename = "rf_sim.in.h5"

gridID = 0

electrodes  =  nses.getSimulationInfo(result_filename,  'electrodes')
print(len(electrodes))
x,  y,  z  =  nses.getSimulationInfo(result_filename,  'grid',  gridID)

print(x)
print(y)
print(z)

for el in electrodes:
    phi  =  nses.getPotentials(result_filename,  gridID,  el)
    print(phi)