from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

RecieveData = 0
if rank == 0:
    data = [(x+1)**x for x in range(size)]
    print('we will be scattering:', data)
else:
    data = None

data = comm.Scatter([data, MPI.INT], [RecieveData, MPI.INT], root=0)
print('rank', rank, 'has data:', data)
