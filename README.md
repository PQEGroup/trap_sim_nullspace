# trap_sim_nullspace
Simulate electric moments of trap chip using nullspace and then compute the voltage sets.

The program works as following:

It first pre process the chip to center it and clean up the gdsfile to save only the electrode information. Then, it creates the journal file for nullspace to generate the model for simulation in abacus file. This requires gds 9.39.0 and not compatible with nullspace python package.

For simulation in nullspace, additional grid file need to be generated first to determine the ion height then the gradient and Hessian of the field. Input deck file will be generated in correspondence of the grid and abacus file that can be run in nullspace.

Finally, output hdf5 is post processed with the input grid all together into a new hdf5 file with clean format for later data processing.