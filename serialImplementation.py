from fileinput import filename
from mpi4py import MPI
import pandas as pd
import numpy as np

filename = input("pls enter file name: ")
df = pd.read_csv(filename)
vectors = df[['x', 'y', 'z']]
magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)

median = np.quantile(magnitudes, 0.5)
Q1 = np.quantile(magnitudes, 0.25)
Q3 = np.quantile(magnitudes, 0.75)
IQR = Q3 - Q1
upper_fence = Q3 + (1.5 * IQR)
lower_fence = Q1 - (1.5 * IQR)

minimum = np.amin(magnitudes)
maximum = np.amax(magnitudes)

print(
    f'----------MPI-----------\n\nMedian:\t{median} \nQ1:\t{Q1} \nQ3: \t{Q3} \nIQR: \t{IQR}\n Minimum: \t{minimum}\n Maximum: \t{maximum} \nUpper fence: \t{upper_fence}\nLower fence: \t{lower_fence}'
)
