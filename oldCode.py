from mpi4py import MPI
import pandas as pd
import numpy as np
Comm = MPI.COMM_WORLD
my_rank = Comm.Get_rank()
p = Comm.Get_size()

name = MPI.Get_processor_name()
global_means = []
lower_quartiles = []
upper_quartiles = []
sorted_mags = []
minimums = []
maximums = []

if my_rank == 0:
    df = pd.read_csv('Accelerometer.csv')
    split_dfs = np.array_split(df, p-1)

    for i in range(0, p-1):
        Comm.send(split_dfs[i], dest=i+1)

if my_rank != 0:
    my_df = Comm.recv(source=0)
    vectors = my_df[['x', 'y', 'z']]
    magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)
    print(magnitudes)
    stats = {}
    stats['lower_quartile'] = np.quantile(magnitudes, 0.25)
    stats['upper_quartile'] = np.quantile(magnitudes, 0.75)
    stats['mean'] = magnitudes.mean()
    stats['minimum'] = np.amin(magnitudes)
    stats['maximum'] = np.amax(magnitudes)

    Comm.send(stats, dest=0)
else:
    for procid in range(1, p):
        stats = Comm.recv(source=procid)
        global_means.append(stats['mean'])
        lower_quartiles.append(stats['lower_quartile'])
        upper_quartiles.append(stats['upper_quartile'])
        minimums.append(stats['minimum'])
        maximums.append(stats['maximum'])

    Q3 = np.quantile(upper_quartiles, 0.75)
    Q1 = np.quantile(lower_quartiles, 0.25)
    IQR = Q3 - Q1
    upper_fence = Q3 + (1.5 * IQR)
    lower_fence = Q1 - (1.5 * IQR)
    global_means = np.array(global_means).mean()
    global_min = np.amin(minimums)
    global_max = np.amax(maximums)
    print(global_max)
