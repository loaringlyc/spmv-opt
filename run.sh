#!/bin/bash
#BSUB -n 1
#BSUB -W 30
#BSUB -q short_gpu
#BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o out.%J
#BSUB -e err.%J
module load PrgEnv-gnu
module load cuda

nvcc spmv.cu -o spmv -lcusparse
./spmv -f test
