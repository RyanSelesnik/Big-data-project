#!/bin/sh
mpiexec -hostfile hostfile -n 1 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 2 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 4 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 8 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 10 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 12 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 16 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 32 python3 -m mpi4py $1 $2
mpiexec -hostfile hostfile -n 64 python3 -m mpi4py $1 $2