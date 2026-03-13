import numpy as np
from nses import *
import  math
from  matplotlib  import  pyplot as plt
from  mpl_toolkits  import  mplot3d as plt3d

x_scan = [0]
y_scan = [0]
z_scan = np.linspace(45,55,101)
filename = 'z_scan.h5'

x_grid = np.array(x_scan)
y_grid = np.array(y_scan)
z_grid = np.array(z_scan)

x_grid = x_grid*1e-6
y_grid = y_grid*1e-6
z_grid = z_grid*1e-6

nses.writeGrid(filename, 0, '3DRECTMESH', x_grid, y_grid, z_grid, True)
