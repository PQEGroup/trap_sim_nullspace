import nses
import h5py

def post_process(result_filename="rf_sim.in.h5", output_filename="rf_sim_out.h5", gridID=0, verbose=True):
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


if __name__ == "__main__":
    post_process()
