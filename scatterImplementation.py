from fileinput import filename
from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

if rank == 0:
    # filename = input('Enter file name: ')
    # time_range = input('Enter ')
    filename = 'Accelerometer.csv'

    df = pd.read_csv(filename)
    chunks = np.array_split(df, numprocs)
else:
    chunks = None

chunk = comm.scatter(chunks, root=0)
vectors = chunk[['x', 'y', 'z']]
magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)

stats = {}
stats['median'] = np.quantile(magnitudes, 0.5)
gathered_chunks = comm.gather(stats, root=0)

if rank == 0:
    medians = []
    for chunk in gathered_chunks:
        medians.append(chunk['median'])

    print(np.quantile(medians, 0.5))
    # median_of_medians()
