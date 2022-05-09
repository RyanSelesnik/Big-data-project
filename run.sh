#!/bin/sh
mpiexec -hostfile hostfile -n $1 python3 -m mpi4py main.py $2