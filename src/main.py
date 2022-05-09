from mpi4py import MPI
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from helperFunctions import getQuartile, preprocess_data
from helperFunctions import get_epochs
from helperFunctions import truncate_df_with_interval
from helperFunctions import write_to_csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()
all_outliers = []

if rank == 0:
    answer = input(
        'Run statistical analysis on entire file (1) or for a given time range (2): ')
    global_start_time = MPI.Wtime()
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    if answer == '2':
        start_date = str(input("Enter start time (yyyy-mm-dd hh:mm:ss): "))
        end_date = str(input("Enter end time (yyyy-mm-dd hh:mm:ss): "))
        epoch_start, epoch_end = get_epochs(start_date, end_date)
        truncate_df_with_interval(df, epoch_start, epoch_end)
    if df.empty:
        raise ValueError('Range does not exist in specified dataset')
    preprocess_data(df)
    chunks = np.array_split(df, numprocs)
else:
    chunks = None

chunk = comm.scatter(chunks, root=0)
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
    output_file_name = 'statistical_indicators.csv'
    write_to_csv(statistical_indicators, output_file_name)

    print(f'\n\n---- Statistcal indicators for {filename} ----\n\n')
    print(
        f'Number of processes used: {numprocs}\n\nMedian:\t{median} \nQ1:\t{Q1} \nQ3: \t{Q3} \nIQR: \t{IQR}\nMinimum: \t{minimum}\nMaximum: \t{maximum} \nUpper fence: \t{upper_fence}\nLower fence: \t{lower_fence}'
    )
    print(f'\nThe above data has been written to ./{output_file_name}')
    print(f'\nThe total execution time is {total_time}')

    vectors = df[['x', 'y', 'z']]
    magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)
    for i in magnitudes:
        if i > upper_fence or i < lower_fence:
            all_outliers.append(i)

    stat = [{
        "label": "Summary",
        "med": median,
        "q1": Q1,
        "q3": Q3,
        "whislo": minimum,
        "whishi": maximum,
        "fliers": all_outliers
    }]

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.bxp(stat, showfliers=True)
    axes.set_label('Magnitude')
    axes.set_yscale('log')
    plt.savefig('box_plot.png')
