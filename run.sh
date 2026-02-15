#!/bin/bash
#BSUB -n 1
#BSUB -W 30
#BSUB -q short_gpu
#BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o out.%J
#BSUB -e err.%J
module load cuda

nvcc spmv.cu -o spmv -lcusparse
./spmv -f shyy41

nsys profile --stats=true ./spmv -f shyy41
# ncu --target-processes=all --set full --launch-count 1 ./spmv -f shyy41
ncu --set summary --launch-count 1 ./spmv -f shyy41
ncu --metrics gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_srcunit_tex_queries_hit_rate.avg.pct,sm__warp_issue_stalled_long_scoreboard_per_warp_active.pct --launch-count 1 ./spmv -f shyy41
