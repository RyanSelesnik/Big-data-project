from fileinput import filename
from mpi4py import MPI
import pandas as pd
import numpy as np

# filename = input("pls enter file name: ")


def getQuartile(arr, quartile):
    n = len(arr)
    if n % 2 == 0:
        return (arr[int(n*quartile) - 1] + arr[int(n*quartile)])/2
    else:
        return arr[int(n*quartile)]


filename = './Accelerometer.csv'
df = pd.read_csv(filename)
vectors = df[['x', 'y', 'z']]
magnitudes = np.apply_along_axis(np.linalg.norm, 1, vectors)
magnitudes.sort()
median = getQuartile(magnitudes, 0.5)
Q1 = getQuartile(magnitudes, 0.25)
Q3 = getQuartile(magnitudes, 0.75)
IQR = Q3 - Q1
upper_fence = Q3 + (1.5 * IQR)
lower_fence = Q1 - (1.5 * IQR)

minimum = np.amin(magnitudes)
maximum = np.amax(magnitudes)

print(
    f'----------Serial-----------\n\nMedian:\t{median} \nQ1:\t{Q1} \nQ3: \t{Q3} \nIQR: \t{IQR}\n Minimum: \t{minimum}\n Maximum: \t{maximum} \nUpper fence: \t{upper_fence}\nLower fence: \t{lower_fence}'
)
