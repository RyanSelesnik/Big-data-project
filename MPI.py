from mpi4py import MPI
import pandas as pd
import numpy as np

time1 = MPI.Wtime()

Comm = MPI.COMM_WORLD
my_rank = Comm.Get_rank()
p = Comm.Get_size()

name = MPI.Get_processor_name()
global_means = []
sorted_mags = []

if my_rank == 0:
    df = pd.read_csv('Accelerometer.csv')
    split_dfs = np.array_split(df, p-1)
    for i in range(0, p-1):
        Comm.send(split_dfs[i], dest=i+1)

if my_rank != 0:
    my_df = Comm.recv(source=0)
    vectors = my_df[['x', 'y', 'z']]
    magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)
    stats = {}
    stats['mean'] = magnitudes.mean()
    Comm.send(stats, dest=0)
else:
    for procid in range(1, p):
        stats = Comm.recv(source=procid)
        global_means.append(stats['mean'])

    global_means = np.array(global_means).mean()
    print(global_means)

time2 = MPI.Wtime()

