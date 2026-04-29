def post_process(result_filename="rf_sim.in.h5", output_filename="rf_sim_out.h5", gridID=0, verbose=True):
    import nses
    import h5py
    

    electrodes = nses.getSimulationInfo(result_filename, "electrodes")
    x, y, z = nses.getSimulationInfo(result_filename, "grid", gridID)
    #electrodes = [22, 23, 24, 25, 27]
    if verbose:
        print(len(electrodes))
        print(x[:, 0, 0])
        print(y[0, :, 0])
        print(z[0, 0, :])

    with h5py.File(output_filename, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("z", data=z)
        for el in electrodes:
            phi = nses.getPotentials(result_filename, gridID, el)
            f.create_dataset(str(el), data=phi)

    print(output_filename)
    return output_filename

def combine_processed(filenames, output_filename="combined_output.h5"):
    import h5py
    import numpy as np

    total_potentials = {}
    X, Y, Z, potentials = read_post_processed(filenames[0])  # Assume all files have the same grid
    total_potentials.update(potentials)

    for filename in filenames[1:]:
        x, y, z, potentials = read_post_processed(filename)
        if not (np.array_equal(X, x) and np.array_equal(Y, y) and np.array_equal(Z, z)):
            raise ValueError(f"Grid mismatch in file {filename}")
        else:
            total_potentials.update(potentials)
    
    sorted_total_dict = dict(sorted(total_potentials.items()))

    with h5py.File(output_filename, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("z", data=z)
        for el, phi in sorted_total_dict.items():
            f.create_dataset(str(el), data=phi)
    
    print(f"Combined output written to {output_filename}")
    return output_filename
    

def read_post_processed(filename="rf_sim_out.h5", electrode_list=None):

    import h5py

    with h5py.File(filename, "r") as f:
        x = f["x"][:]
        y = f["y"][:]
        z = f["z"][:]

        if electrode_list is not None:
            electrodes = electrode_list
        else:
            electrodes = [key for key in f.keys() if key not in ["x", "y", "z"]]

        # Datasets are stored under string keys, but callers may pass integer IDs.
        potentials = {el: f[str(el)][:] for el in electrodes}

    return x, y, z, potentials

def save_to_artiq(Vks, electrode, mapping_dict, filename="artiq_output.csv"):

    import numpy as np

    input_num = mapping_dict.get("artiq_input_num")
    if Vks.ndim == 1:
        output_num = np.zeros((input_num, 1), dtype=np.float64)
        for el, voltage in zip(electrode, Vks):
            output_num[mapping_dict[el] - 1, 0] = voltage

        with open(filename, "w") as f:
            for i in range(output_num.shape[0]):
                f.write(", ".join([f"{output_num[i, j]:.6f}" for j in range(output_num.shape[1])]) + "\n")

    elif Vks.ndim == 2:
        output_num = np.zeros((Vks.shape[0], input_num), dtype=np.float64)
        for i in range(Vks.shape[0]):
            for el, voltage in zip(electrode, Vks[i]):
                output_num[i, mapping_dict[el]-1] = voltage
        #print(output_num.shape)

        with open(filename, "w") as f:
            for i in range(output_num.shape[0]):
                f.write(", ".join([f"{output_num[i, j]:.6f}" for j in range(output_num.shape[1])]) + "\n")


def read_from_artiq(electrode, mapping_dict, filename="artiq_output.csv"):
    import numpy as np

    output_num = np.loadtxt(filename, delimiter=",", ndmin=2)

    if output_num.shape[1] == 1:
        Vks = np.array(
            [output_num[mapping_dict[el] - 1, 0] for el in electrode],
            dtype=np.float64,
        )
    else:
        Vks = np.array(
            [
                [output_num[i, mapping_dict[el] - 1] for el in electrode]
                for i in range(output_num.shape[0])
            ],
            dtype=np.float64,
        )

    return Vks

if __name__ == "__main__":
    post_process()
