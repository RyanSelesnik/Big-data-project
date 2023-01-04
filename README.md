# Sensor Measurement Analysis for a Large Data Set Using MPI in Python

### Contributors: Jamie Benater, Rachel Edelstein, Ryan Selesnik

This program computes statistical indicators on a given data set, generated by a sensor in an Inertial Measurement Unit (IMU) device. The program uses MPI in Python, using the mpi4py library, to complete the task using parallel techniques to improve efficiency. The statistical indicators calculated in the program are, namely: median, minimum, maximum, both upper and lower quartiles, as well as outliers. These are used to generate a box-and-whisker plot. 

The user is required to specify/provide a data set and has the option to provide a time range on which to perform the statistical analysis.

## Compiling and Running

It is assumed that this code will be run on the University of The Witwatersrand computer cluster.

To run the code with the example data set given in `/data/Accelerometer.csv`:
Ensure you're in the root directory and use the following command:

```
./run.sh 10 ./data/Accelerometer.csv
```
This will prompt you with two options, (1) and (2). Choose (1) to run the entire file or (2) to specify a time range, where the start date and end date must have the following format 

```
(yyyy-mm-dd hh:mm:ss)
```

This will run the code with 10 processes using the nodes specified in `./hostfile`. Furthermore, the results are written to `./statistical_indicators.csv`

Note, to ensure the run script is executable, run the following:

```
chmod +x run.sh
```


