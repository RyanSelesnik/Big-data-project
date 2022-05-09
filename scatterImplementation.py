from fileinput import filename
from mpi4py import MPI
import pandas as pd
import numpy as np
from helperFunctions import getQuartile
from helperFunctions import get_epochs
from helperFunctions import truncate_df_with_interval
from helperFunctions import write_to_csv

import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

if rank == 0:
    global_start_time = MPI.Wtime()
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    chunks = np.array_split(df, numprocs)
else:
    chunks = None
    
chunk = comm.scatter(chunks, root=0)
# Change to last three columns
vectors = chunk[['x', 'y', 'z']]
magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)
magnitudes.sort()

# Get median, quartiles, min and max magnitudes of each chunk
stats = {}
stats['median'] = getQuartile(magnitudes, 0.5)
stats['q1'] = getQuartile(magnitudes, 0.25)
stats['q3'] = getQuartile(magnitudes, 0.75)
stats['minimum'] = np.amin(magnitudes)
stats['maximum'] = np.amax(magnitudes)
gathered_chunks = comm.gather(stats, root=0)

if rank == 0:
    medians = []
    Q3s = []
    Q1s = []
    minimums = []
    maximums = []
    for chunk in gathered_chunks:
        medians.append(chunk['median'])
        Q3s.append(chunk['q3'])
        Q1s.append(chunk['q1'])
        minimums.append(stats['minimum'])
        maximums.append(stats['maximum'])

    medians.sort()
    Q1s.sort()
    Q3s.sort()
    median = getQuartile(medians, 0.5)
    Q1 = getQuartile(Q1s, 0.25)
    Q3 = getQuartile(Q3s, 0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + (1.5 * IQR)
    lower_fence = Q1 - (1.5 * IQR)

    minimum = np.amin(minimums)
    maximum = np.amax(maximums)

    statistical_indicators = {
        'file_name': filename,
        'lower_quartile': Q1,
        'median': median,
        'upper_quartile': Q3,
        'interquartile_range': IQR,
        'minimum': minimum,
        'maximum': maximum,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence
    }
    total_time = MPI.Wtime() - global_start_time
    write_to_csv(statistical_indicators)

    print(
        f'----------MPI-----------\n\n Number of processes: {numprocs}\nMedian:\t{median} \nQ1:\t{Q1} \nQ3: \t{Q3} \nIQR: \t{IQR}\n Minimum: \t{minimum}\n Maximum: \t{maximum} \nUpper fence: \t{upper_fence}\nLower fence: \t{lower_fence}'
    )
    print(f'Final time is {total_time}')
