import dask.dataframe as dd
from fileinput import filename
from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

if rank == 0:
    filename = 'Accelerometer.csv'
    df = pd.read_csv(filename)
    chunks = np.array_split(df, numprocs)
else:
    chunks = None

chunk = comm.scatter(chunks, root=0)
vectors = chunk[['x', 'y', 'z']]
magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)

# Get median and quartiles of each chunk
stats = {}
stats['median'] = np.quantile(magnitudes, 0.5)
stats['q1'] = np.quantile(magnitudes, 0.25)
stats['q3'] = np.quantile(magnitudes, 0.75)
gathered_chunks = comm.gather(stats, root=0)

if rank == 0:
    medians = []
    Q3s = []
    Q1s = []
    for chunk in gathered_chunks:
        medians.append(chunk['median'])
        Q3s.append(chunk['q3'])
        Q1s.append(chunk['q1'])

    median = np.quantile(medians, 0.5)
    Q1 = np.quantile(Q1s, 0.25)
    Q3 = np.quantile(Q3s, 0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + (1.5 * IQR)
    lower_fence = Q1 - (1.5 * IQR)
    print(
        f'----------MPI-----------\n\nMedian:\t{median} \nQ1:\t{Q1} \nQ3: \t{Q3} \nIQR: \t{IQR}\nUpper fence: \t{upper_fence}\nLower fence: \t{lower_fence}'
    )


