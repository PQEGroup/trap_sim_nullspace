def post_process(result_filename="rf_sim.in.h5", output_filename="rf_sim_out.h5", gridID=0, verbose=True):
    import nses
    import h5py

    electrodes = nses.getSimulationInfo(result_filename, "electrodes")
    x, y, z = nses.getSimulationInfo(result_filename, "grid", gridID)

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


if __name__ == "__main__":
    post_process()
