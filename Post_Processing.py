from nses import *
from matplotlib import pyplot as plt
import os
import pandas as pd

os.chdir('/home/zx296/Desktop/trap_sim_nullspace')

#data_folder_axial = os.path.join(os.getcwd(), 'nullspace_sim_result_axial')
#data_folder_tilt =  os.path.join(os.getcwd(), 'nullspace_sim_result_tilt')
tst_folder = os.path.join(os.getcwd(), 'tst_result')

print(os.getcwd())
if not os.path.exists(data_folder_axial):
    os.mkdir(data_folder_axial)
if not os.path.exists(data_folder_tilt):
    os.mkdir(data_folder_tilt)
if not os.path.exists(tst_folder):
    os.mkdir(tst_folder)

gridID = 0

electrodes  =  nses.getSimulationInfo(result_filename,  'electrodes')
print(len(electrodes))
x,  y,  z  =  nses.getSimulationInfo(result_filename,  'grid',  gridId)

print(x)
print(y)
print(z)

for el in electrodes:
    phi  =  nses.getPotentials(result_filename,  gridId,  el)
    print(phi)