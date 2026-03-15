import numpy as np
import nses


def generate_grid_file(x_scan=None, y_scan=None, z_scan=None, filename="z_scan.h5"):
	if x_scan is None:
		x_scan = [0]
	if y_scan is None:
		y_scan = [0]
	if z_scan is None:
		z_scan = np.linspace(45, 55, 101)

	x_grid = np.array(x_scan) * 1e-6
	y_grid = np.array(y_scan) * 1e-6
	z_grid = np.array(z_scan) * 1e-6

	nses.writeGrid(filename, 0, "3DRECTMESH", x_grid, y_grid, z_grid, True)
	print(filename)
	return filename


if __name__ == "__main__":
	generate_grid_file()
