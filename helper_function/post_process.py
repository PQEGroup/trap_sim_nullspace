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


def plot_voltage_set_on_gds(
    voltage_set,
    mapping_data_file="mapping_data.pkl",
    gds_file="chip_design_merged_flattened.gds",
    target_layer=(1, 0),
    set_index=-1,
    title="Final voltage set",
    save_path=None,
    show=True,
    annotate_values=True,
    interactive=False,
    polygon_order="gds",
    electrode_order=None,
):
    import pickle
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.path import Path as MplPath

    import gdsfactory as gf
    from gdsfactory.gpdk import get_generic_pdk

    def _load_voltage_array(source):
        if isinstance(source, (str, Path)):
            values = np.loadtxt(source, delimiter=",", ndmin=2)
        else:
            values = np.asarray(source)

        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        elif values.ndim == 2:
            values = values[set_index]

        return np.asarray(values, dtype=np.float64).reshape(-1)

    def _format_voltage(value):
        text = f"{value:+.3f}".rstrip("0").rstrip(".")
        return text if text not in {"+", "-"} else f"{value:+.3f}"

    def _parse_electrode_token(item):
        if item == 0:
            return None
        if isinstance(item, (int, np.integer)):
            return int(item)
        if isinstance(item, str):
            if "_" in item:
                suffix = item.split("_")[-1]
                if suffix.isdigit():
                    return int(suffix)
            if item.isdigit():
                return int(item)
        return None

    def _centroid(points):
        return points.mean(axis=0)

    def _order_polygon_indices(points_list, n_slots, mode):
        centroids = np.array([_centroid(p) for p in points_list], dtype=np.float64)
        n_use = min(len(points_list), n_slots)

        if mode == "gds":
            return list(range(n_use))

        # Slot-aware geometric ordering:
        # top row (left -> right) then bottom row (left -> right).
        if n_use % 2 == 0 and n_use >= 4:
            row_width = n_use // 2
            y_desc = np.argsort(-centroids[:, 1])[:n_use]
            top_pool = y_desc[:row_width]
            bot_pool = y_desc[row_width:2 * row_width]
            top_sorted = top_pool[np.argsort(centroids[top_pool, 0])]
            bot_sorted = bot_pool[np.argsort(centroids[bot_pool, 0])]
            return list(top_sorted) + list(bot_sorted)

        # Fallback: simple row-major ordering by y then x.
        ordered = np.lexsort((centroids[:, 0], -centroids[:, 1]))
        return list(ordered[:n_use])

    get_generic_pdk().activate()

    with open(mapping_data_file, "rb") as f:
        mapping_data = pickle.load(f)

    mapping_list = list(mapping_data["mapping_list"])
    mapping_dict = mapping_data.get("mapping_dict", {})
    voltage_values = _load_voltage_array(voltage_set)

    slot_count = len(mapping_list)
    slot_values = np.zeros(slot_count, dtype=np.float64)
    slot_electrodes = [_parse_electrode_token(item) for item in mapping_list]
    active_electrodes = [el for el in slot_electrodes if el is not None]
    artiq_input_num = int(mapping_dict.get("artiq_input_num", slot_count))

    if voltage_values.size == slot_count or voltage_values.size == artiq_input_num:
        # Input is a slot-ordered vector (ARTIQ channel order).
        if voltage_values.size < slot_count:
            raise ValueError(
                f"Voltage set has {voltage_values.size} slot entries but mapping_list expects {slot_count}."
            )
        slot_values[:] = voltage_values[:slot_count]
    elif voltage_values.size == len(active_electrodes):
        # Input is an electrode-ordered vector. Convert electrode -> slot using mapping_dict.
        if electrode_order is None:
            electrode_order = sorted(active_electrodes)

        if len(electrode_order) != voltage_values.size:
            raise ValueError(
                "electrode_order length does not match active voltage count."
            )

        active_lookup = {
            int(electrode): value
            for electrode, value in zip(electrode_order, voltage_values)
        }
        for electrode, value in active_lookup.items():
            slot_index = mapping_dict.get(electrode, None)
            if slot_index is not None:
                slot_index = int(slot_index) - 1
                if 0 <= slot_index < slot_count:
                    slot_values[slot_index] = value

        # Fallback for mappings missing in mapping_dict: place by mapping_list parsing.
        for index, electrode in enumerate(slot_electrodes):
            if electrode is not None and np.isclose(slot_values[index], 0.0):
                slot_values[index] = active_lookup.get(electrode, 0.0)
    else:
        raise ValueError(
            "Voltage set length does not match the saved slot count or the active electrode count."
        )

    component = gf.import_gds(str(gds_file))
    polygons_by_layer = component.get_polygons_points(by="tuple", layers=[target_layer])
    polygons = polygons_by_layer.get(target_layer, [])
    if not polygons:
        raise ValueError(f"No polygons found on layer {target_layer} in {gds_file}")

    polygon_arrays = [np.asarray(points, dtype=np.float64) for points in polygons]
    if polygon_order != "gds":
        ordered_idx = _order_polygon_indices(polygon_arrays, len(polygon_arrays), polygon_order)
        polygon_arrays = [polygon_arrays[i] for i in ordered_idx]

    x_min = min(points[:, 0].min() for points in polygon_arrays)
    x_max = max(points[:, 0].max() for points in polygon_arrays)
    y_min = min(points[:, 1].min() for points in polygon_arrays)
    y_max = max(points[:, 1].max() for points in polygon_arrays)

    span_x = x_max - x_min
    span_y = y_max - y_min

    # Build electrode voltages from slot voltages: slot -> electrode.
    electrode_values = {}
    for slot_index, electrode in enumerate(slot_electrodes):
        if electrode is not None and slot_index < slot_values.size:
            electrode_values[electrode] = slot_values[slot_index]

    # Electrode number is the raw GDS polygon index (1-based).
    polygon_electrodes = list(range(1, len(polygon_arrays) + 1))
    polygon_values = np.array(
        [electrode_values.get(electrode_num, 0.0) for electrode_num in polygon_electrodes],
        dtype=np.float64,
    )

    max_abs = np.max(np.abs(polygon_values)) if np.any(polygon_values) else 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(16, 6))
    patches = []
    for electrode_num, points, value in zip(polygon_electrodes, polygon_arrays, polygon_values):
        facecolor = "white" if np.isclose(value, 0.0) else cmap(norm(value))
        patch = ax.fill(
            points[:, 0],
            points[:, 1],
            facecolor=facecolor,
            edgecolor="0.3",
            linewidth=0.8,
            alpha=0.95,
        )[0]
        patches.append((patch, electrode_num, points, value))

        if annotate_values and not np.isclose(value, 0.0):
            centroid = _centroid(points)
            text = _format_voltage(value)
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(
                centroid[0],
                centroid[1],
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    pad_x = span_x * 0.05 if span_x > 0 else 1.0
    pad_y = span_y * 0.08 if span_y > 0 else 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Voltage (V)")

    if interactive:
        def on_click(event):
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return
            for patch, electrode_num, points, value in patches:
                closed = np.vstack([points, points[0]])
                if MplPath(closed).contains_point((event.xdata, event.ydata)):
                    patch.set_edgecolor("black")
                    patch.set_linewidth(1.2)
                    fig.canvas.draw_idle()
                    break

        fig.canvas.mpl_connect("button_press_event", on_click)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

if __name__ == "__main__":
    post_process()
