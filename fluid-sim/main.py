#!/bin/env python3

import tensorflow as tf
import numpy as np

# Create fluid grid (will start with hard-coded 512x512 2D grid)
# Incompressible fluid: fluid state boils down to a velocity field (3 variables)
N = 512 # Number of cells on each side of simulation grid
u = np.empty((N, N, 3)) # An NxN grid of fluid velocity 3-vectors

# Initialize grid
for i in range(N):
    for j in range(N):
        if ((i-(N/2.))**2 + (j-(N/2.))**2) < (N/2.)**2:
            u[i, j][0] = 1.
        #
    #
#
# Phil: do we want to be the type of people who add little '#' to make up for python's confusing lack of close-brackets? (see above)

# Establish operations to update grid at each timestep (see Nvidia tutorail)
# (Don't forget boundary conditions!)

# Loop and apply timestep operations repeatedly, outputting state of grid after every nth update
# Optionally, use the velocity field to advect some other variable at each timestep, to visualize fluid flow, and output this every nth update as well

# Collect nobel prize
