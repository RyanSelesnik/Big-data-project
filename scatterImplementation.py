from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()


if rank == 0:
    df = pd.read_csv('Accelerometer.csv')
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
    for chunk in gathered_chunks:
        print(chunk)
