from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD


MPI.File.Get_size('./Accelerometer.csv')
