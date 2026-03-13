from nses import *
from matplotlib import pyplot as plt
import os
import pandas as pd

os.chdir('/home/zx296/Desktop/Trap_sim/Test1')

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

def obtain_moment(axis, type='axial'):
    res=101
    if axis == 'x' and type=='axial':
        result_filename='SET_X_deck.in.h5'
        gridId  =  0
    elif axis == 'y' and type=='axial':
        result_filename='SET_Y_deck.in.h5'
        gridId  =  1
    elif axis == 'z' and type=='axial':
        result_filename='SET_Z_deck.in.h5'
        gridId  =  2
    elif axis == 'x' and type=='tilt':
        result_filename='test.in.h5'
        gridId  =  4
    elif axis == 'y' and type=='tilt':
        result_filename='SET_Y_tilt_deck.in.h5'
        gridId  =  5
    elif axis == 'z' and type=='tilt':
        result_filename='SET_Z_tilt_deck.in.h5'
        gridId  =  6

    electrodes  =  nses.getSimulationInfo(result_filename,  'electrodes')
    print(len(electrodes))
    x,  y,  z  =  nses.getSimulationInfo(result_filename,  'grid',  gridId)
    

#'''
    for el in electrodes:
        if type=='axial':
            electrode_folder = os.path.join(tst_folder, str(el))
        elif type=='tilt':
            electrode_folder = os.path.join(tst_folder, str(el))

        if not os.path.exists(electrode_folder):
            os.mkdir(electrode_folder)
        if axis == 'x' and type=='axial':
            filename='Electrode_' + str(el) + '_X_moment'
            data = np.empty((101, 2))
            phi  =  nses.getPotentials(result_filename,  gridId,  el)
            data[:, 0] = x[:, 0, 0]/1e-6
            data[:, 1] = phi[:, 0, 0]
            columns = ['x', 'V']
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(os.path.join(electrode_folder, filename),index=False)
        elif axis == 'y' and type=='axial':
            for i, p in enumerate(x[:, 0, 0]):
                filename='Electrode_' + str(el) + '_Y_moment_at_x=' + str(p/1e-6) + '_um'
                data = np.empty((101, 2))
                phi  =  nses.getPotentials(result_filename,  gridId,  el)
                data[:, 0] = y[i, :, 0]/1e-6
                data[:, 1] = phi[i, :, 0]
                columns = ['y', 'V']
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(electrode_folder, filename),index=False)
        elif axis =='z' and type=='axial':
            for i, p in enumerate(x[:, 0, 0]):
                filename='Electrode_' + str(el) + '_Z_moment_at_x=' + str(p/1e-6) + '_um'
                data = np.empty((101, 2))
                phi  =  nses.getPotentials(result_filename,  gridId,  el)
                data[:, 0] = z[i, 0, :]/1e-6
                data[:, 1] = phi[i, 0, :]
                columns = ['z', 'V']
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(electrode_folder, filename),index=False)
        elif axis == 'x' and type=='tilt':
            tx = np.linspace(-100, 100, res) * 1e-6
            for i in range(res):
                filename='Electrode_' + str(el) + '_X_moment_at_x=' + str(tx[i]/1e-6) + '_um'
                data = np.empty((101, 2))
                phi  =  nses.getPotentials(result_filename,  gridId,  el)
                data[:, 0] = x[i*res:(i+1)*res]/1e-6
                data[:, 1] = phi[i*res:(i+1)*res, 0]
                columns = ['x', 'V']
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(electrode_folder, filename),index=False)
        elif axis == 'y' and type=='tilt':
            for i in range(res):
                filename='Electrode_' + str(el) + '_Y_moment_at_x=' + str(x[i*res]/1e-6) + '_um'
                data = np.empty((101, 2))
                phi  =  nses.getPotentials(result_filename,  gridId,  el)
                data[:, 0] = y[i*res:(i+1)*res]/1e-6
                data[:, 1] = phi[i*res:(i+1)*res, 0]
                columns = ['y', 'V']
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(electrode_folder, filename),index=False)
        elif axis == 'z' and type=='tilt':
            for i in range(res):
                filename='Electrode_' + str(el) + '_Z_moment_at_x=' + str(x[i*res]/1e-6) + '_um'
                data = np.empty((101, 2))
                phi  =  nses.getPotentials(result_filename,  gridId,  el)
                data[:, 0] = z[i*res:(i+1)*res]/1e-6
                data[:, 1] = phi[i*res:(i+1)*res, 0]
                columns = ['z', 'V']
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(electrode_folder, filename),index=False)
#'''


def plot_moment(axis, pos=None, type='axial'):
    x_string='X_moment'
    y_string='Y_moment'
    z_string='Z_moment'

    if type=='axial':
        data_folder = tst_folder
    elif type=='tilt':
        data_folder = data_folder_tilt

    if axis=='x':
        addr =[]
        for dir in os.listdir(data_folder):
            filex = [file for file in os.listdir(os.path.join(data_folder, dir)) if x_string in file]
            addr.append(os.path.join(data_folder, dir + '/' + filex[0]))
        addr.sort(key = lambda file: float(file.split('/')[-1].split('_')[1]))
        num_plt = len(addr)
        data = np.empty((len(addr), 101))

        for i in range(num_plt):
            df = pd.read_csv(addr[i])
            data[i,:]=df['V'].values

        x = pd.read_csv(addr[0])['x'].values
        fig, ax =plt.subplots(figsize=(4, 3))
        cmap = plt.get_cmap('prism')

        for i in range(0, 10):
            ax.plot(x, data[i, :], color = cmap(i*2), label='Electrode' + str(i + 1))

        ax.set_xlabel('$x$  ($\\mu$m)')
        ax.set_ylabel('$Potential$  (V)')
        #ax.set_title('Electrode moment along  x-axis')
        ax.legend(fontsize='small')
        fig.savefig('ionTraptst.png',  transparent=False,  bbox_inches='tight')
    elif axis =='y':
        addr =[]
        for dir in os.listdir(data_folder):
            filey = [file for file in os.listdir(os.path.join(data_folder, dir)) if y_string in file and str(pos) in file]
            filey.sort(key = lambda file: np.abs(pos - float(file.split('_')[-2][2:])))
            addr.append(os.path.join(data_folder, dir + '/' + filey[0]))
        addr.sort(key = lambda file: float(file.split('/')[-1].split('_')[1]))
        num_plt = len(addr)
        data = np.empty((len(addr), 101))

        for i in range(num_plt):
            df = pd.read_csv(addr[i])
            data[i,:]=df['V'].values

        x = pd.read_csv(addr[0])['y'].values
        fig, ax =plt.subplots(figsize=(15, 7))
        cmap = plt.get_cmap('prism')

        for i in range(num_plt-2):
            ax.plot(x, data[i, :], color = cmap(i), label='Electrode' + str(i + 1))

        ax.set_xlabel('$y$  ($\\mu$m)')
        ax.set_ylabel('$Potential$  (V)')
        ax.set_title('Electrode moment along y-axis at x = {} $\\mu$m'.format(pos))
        ax.legend()
        fig.savefig('ionTrapY.png',  transparent=False,  bbox_inches='tight')
    elif axis == 'z':
        addr =[]
        for dir in os.listdir(data_folder):
            filez = [file for file in os.listdir(os.path.join(data_folder, dir)) if z_string in file and str(pos) in file]
            filez.sort(key = lambda file: np.abs(pos - float(file.split('_')[-2][2:])))
            addr.append(os.path.join(data_folder, dir + '/' + filez[0]))
        addr.sort(key = lambda file: float(file.split('/')[-1].split('_')[1]))
        num_plt = len(addr)
        data = np.empty((len(addr), 101))

        for i in range(num_plt):
            df = pd.read_csv(addr[i])
            data[i,:]=df['V'].values

        x = pd.read_csv(addr[0])['z'].values
        fig, ax =plt.subplots(figsize=(15, 7))
        cmap = plt.get_cmap('prism')

        for i in range(num_plt-2):
            ax.plot(x, data[i, :], color = cmap(i), label='Electrode' + str(i + 1))

        ax.set_xlabel('$z$  ($\\mu$m)')
        ax.set_ylabel('$Potential$  (V)')
        ax.set_title('Electrode moment along z-axis at x = {} $\\mu$m'.format(pos))
        ax.legend()
        fig.savefig('ionTrapZ.png',  transparent=False,  bbox_inches='tight')

#obtain_moment('x')
#obtain_moment('y')
obtain_moment('z')
#obtain_moment('x', type='tilt')
#obtain_moment('y', type='tilt')
#obtain_moment('z', type='tilt')

#plot_moment('x')
#plot_moment('y', 62)
#plot_moment('z', 60, 'tilt')


'''
gridId  =  0
x,  y,  z  =  nses.getSimulationInfo(result_filename,  'grid',  gridId)

electrodeId  =  6
phi  =  nses.getPotentials(result_filename,  gridId,  electrodeId)

#idxX  =  np.where(np.unique(x)  ==  0)
#yRng  =  np.unique(y)/1e-6
#idxY  =  (yRng  >  -75)  &  (yRng  <  75)
fig  =  plt.figure()

#phiWellyz  =  np.squeeze(phi[idxX,  idxY,  :])
#im  =  plt.imshow(np.transpose(phiWellyz),  interpolation='bilinear', extent  =  [y.min()/1e-6,  y.max()/1e-6,  z.min()/1e-6,  z.max()/1e-6],  origin='lower', vmin  =  phiWellyz.min(),  vmax  =  phiWellyz.max())

plt.plot(x[:, 0, 0]/1e-6, phi[:, 0, 0])
plt.xlabel('$x$  ($\\mu$m)')
plt.ylabel('$Potential$  (V)')
plt.title('Potential  along  x-axis')
plt.grid()

#fig.colorbar(im)
fig.savefig('ionTrapDC.png',  transparent=True,  bbox_inches='tight') 

plt.show(block=True)
'''
