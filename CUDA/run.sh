#!/bin/bash
#SBATCH --job-name=GPU
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
## Use cuda-int from PC labs; otherwise, use cuda-ext.q
#SBATCH --partition=cuda-ext.q
## GPUs are available on aolin cluster
#SBATCH --gres=gpu:GeForceRTX3080:1
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt

set -e
module load nvhpc/21.2
nvcc -o main laplace_skeleton.cu -lcudart # compile
# Run:
nsys nvprof --print-gpu-trace ./main 4096 4096 100 32

#Profiling summary:
#nsys nvprof --print-gpu-trace ./main 1000 1000

#Detailed profiling (Optional):
#ncu --target-processes application-only --set full -f -o profile.ncu-rep ./main 100 100 # This generates a file named "profile.ncu-rep"
# 1. Manually download file "profile.ncu-rep" to your computer
# 2. Download the "Nsys" profiler from the Nvidia website on your computer (https://developer.nvidia.com/nsight-systems).
# 3. Use "ncu" from your computer to open your profile report

# Profile traces with nsys (Optional):
# nsys profile -f true -t nvtx,cuda -o profile.nsys-rep ./main 100 100
# Download "profile.nsys-rep.qdrep" on your computer and use nsys to open the file.

