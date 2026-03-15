mesh_file='chip.inp'# Abaqus file
input_deck_filename='rf_sim.in'
grid_file='z_scan.h5'
electrode_num = 27
rf_electrode_index = [13]
gnd_electrode_index = [2, 14, 26]
#excitation_electrodes = range(1, electrode_num + 1) - set(gnd_electrode_index) - set(rf_electrode_index)
excitation_electrodes = rf_electrode_index

command_tag_dict = {'mesh': 'MESH', 
                'mesh_quality': 'MESH_QUALITY', 
                'electrode': 'ELECTRODE_EXCITATION', 
                'symmetry': 'SYMMETRY',
                'solver': 'SOLVER',
                'clusters':'CLUSTERS',
                'output':'OUTPUT'}

mesh_tag = {'file': 'FILE', 'scale': 'SCALE'}
mesh_quality_tag={'shape_min':'SHAPE_MIN', 
                  'shape_max':'SHAPE_MAX', 
                  'cond_num_min': 'CONDITION_NUMBER_MIN', 
                  'cond_num_max': 'CONDITION_NUMBER_MAX',
                  'shape_sev':'SHAPE_SEVERITY', 
                  'cond_num_sev':'CONDITION_NUMBER_SEVERITY',
                  'report': 'REPORT'}
electrode_exci_tag ={'file':'FILE', 'list':'LIST'}
symmetry_tag={'xy': 'SYMMETRY_PLANE_XY', 
          'xy_offset': 'SYMMETRY_PLANE_XY_OFFSET',
          'xy_width': 'SYMMETRY_PLANE_XY_WIDTH',
          'xy_N': 'SYMMETRY_PLANE_XY_N',
          'xz':'SYMMETRY_PLANE_XZ',
          'xz_offset':'SYMMETRY_PLANE_XZ_OFFSET',
          'yz':'SYMMETRY_PLANE_YZ',
          'yz_offset':'SYMMETRY_PLANE_YZ_OFFSET',
          'rot_mode':'SYMMETRY_ROTATION_MODE',
          'rot_z':'SYMMETRY_ROTATION_Z'}
solver_tag={'precision':'',
        'system': 'SYSTEM',
        'fil_comp_tol':'FILL_COMPRESS_TOLERANCE',
        'sol_comp_tol':'SOLVE_COMPRESS_TOLERANCE',
        'fil_recomp':'FILL_RECOMPRESS',
        'fil_recomp_tol':'FILL_RECOMPRESS_TOLERANCE',
        'comp_off_diag':'COMPRESS_ALL_OFF_DIAGONAL_BLOCKS',
        'aca_res':'ACA_FILL_RESTART',
        'aca_res_size': 'ACA_FILL_RESTART_SIZE',
        'aca_res_pc':'ACA_FILL_RESTART_PERCENT'}
clusters_tag={'max_lvl': 'CLUSTER_MAX_LEVEL', 
              'min_size':'CLUSTER_MINIMAL_SIZE',
              'min_comp_size':'CLUSTER_MINIMAL_COMPRESSION_SIZE',
              'dist_meth':'CLUSTER_DISTANCE_METHOD'}
output_tag = {'vtu':'CHARGE_VTU', 
              'grid':'GRID',
              'pot':'POTENTIAL', 
              'efield': 'ELECTRIC_FIELD',
              'xmf':'XMF'}

# In each command tag group assign values to each tag
# modify the assign value in this dictionary to be written in input deck 
def build_commands(mesh_file, grid_file, excitation_electrodes):
     return {
          'mesh':{mesh_tag['file']: mesh_file,
              mesh_tag['scale']: 1e-6},
          'mesh_quality':{mesh_quality_tag['shape_min']:0.2,
              mesh_quality_tag['shape_max']:10.0,
              mesh_quality_tag['cond_num_min']:1.0,
              mesh_quality_tag['cond_num_max']:3.0,
              mesh_quality_tag['shape_sev']:0,
              mesh_quality_tag['cond_num_sev']:0,
              mesh_quality_tag['report']:'OFF'},
          'electrode':{electrode_exci_tag['file']:None,
              electrode_exci_tag['list']:' '.join(str(i) for i in excitation_electrodes)},
          'symmetry':{symmetry_tag['xy']:'NONE',
              symmetry_tag['xy_offset']:0.0,
              symmetry_tag['xy_width']:0.0,
              symmetry_tag['xy_N']:5,
              symmetry_tag['xz']:'NONE',
              symmetry_tag['xz_offset']:0.0,
              symmetry_tag['yz']:'NONE',
              symmetry_tag['yz_offset']:0.0,
              symmetry_tag['rot_mode']:'NONE',
              symmetry_tag['rot_z']:-1.0},
          'solver':{solver_tag['precision']:None,
              solver_tag['system']:'COMPRESS',
              solver_tag['fil_comp_tol']:None,
              solver_tag['sol_comp_tol']:None,
              solver_tag['fil_recomp']:None,
              solver_tag['fil_recomp_tol']:None,
              solver_tag['comp_off_diag']:None,
              solver_tag['aca_res']:None,
              solver_tag['aca_res_size']:None,
              solver_tag['aca_res_pc']:None},
          'clusters':{clusters_tag['dist_meth']:None,
              clusters_tag['max_lvl']:None,
              clusters_tag['min_comp_size']:None,
              clusters_tag['min_size']:None},
          'output':{output_tag['efield']:'OFF',
              output_tag['grid']:grid_file,
              output_tag['pot']:'ON',
              output_tag['vtu']:'OFF',
              output_tag['xmf']:'OFF'}
     }

# comment out unneccesary operations
command_lst = ['mesh', 
               'mesh_quality',
               'electrode',
               #'symmetry',
               'solver',
               #'cluster',
               'output']

def write_tag(tag, value):
    if value is not None:
        return tag + ' = ' + str(value) + '\n'

def write_tag_group(file, command_tag, commands):
    f = open(file=file,mode='a+')
    f.write('<' + command_tag_dict[command_tag] + '_BEG>' + '\n')
    tags = commands[command_tag]
    for key, val in tags.items():
        if val is not None:
            f.write('  ' + write_tag(key, val))
    f.write('<' + command_tag_dict[command_tag] + '_END>' + '\n')
    f.close()

def write_commands(command_lst, commands, file=input_deck_filename):
    # clear previous commands
    f= open(file, "w")
    f.close()
    
    for command in command_lst:
        write_tag_group(file, command, commands)


def generate_input_deck(
    mesh_file=mesh_file,
    input_deck_filename=input_deck_filename,
    grid_file=grid_file,
    excitation_electrodes=None,
    command_lst=None,
):
    if excitation_electrodes is None:
        excitation_electrodes = rf_electrode_index

    if command_lst is None:
        command_lst = [
            'mesh',
            'mesh_quality',
            'electrode',
            'solver',
            'output',
        ]

    commands = build_commands(mesh_file, grid_file, excitation_electrodes)
    write_commands(command_lst=command_lst, commands=commands, file=input_deck_filename)
    print(input_deck_filename)
    return input_deck_filename


if __name__ == '__main__':
    generate_input_deck(command_lst=command_lst)
