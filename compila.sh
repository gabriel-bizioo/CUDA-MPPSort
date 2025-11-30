#!/bin/bash

host=`hostname`
echo $host
if [ $host = "aura" ]; then
  echo "compilando para maquina aura."
  nvcc  -DTHREADS_PER_BLOCK=1024U -o mpp-sort-bitonic-1024 main.cu -I../NVIDIA_CUDA-10.1_Samples/common/ -I../NVIDIA_CUDA-10.1_Samples/common/inc/

  nvcc  -DTHREADS_PER_BLOCK=512U -o mpp-sort-bitonic-512 main.cu -I../NVIDIA_CUDA-10.1_Samples/common/ -I../NVIDIA_CUDA-10.1_Samples/common/inc/

  nvcc  -DTHREADS_PER_BLOCK=256U -o mpp-sort-bitonic-256 main.cu -I../NVIDIA_CUDA-10.1_Samples/common/ -I../NVIDIA_CUDA-10.1_Samples/common/inc/
fi

if [ $host = "nv00" ]; then
   echo "compilando para maquina nv00."
   nvcc  -DTHREADS_PER_BLOCK=1024U  --std=c++14 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ -o mpp-sort-bitonic-1024 main.cu 

   nvcc  -DTHREADS_PER_BLOCK=512U  --std=c++14 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ -o mpp-sort-bitonic-512 main.cu 

   nvcc  -DTHREADS_PER_BLOCK=256U  --std=c++14 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ -o mpp-sort-bitonic-256 main.cu
fi

if [ $host = "orval" ]; then
   echo "compilando para maquina orval."
   xx
   nvcc  -DTHREADS_PER_BLOCK=1024U  -arch sm_50 --allow-unsupported-compiler  -std=c++17 -Xcompiler=-std=c++17 -ccbin /usr/bin/g++-12 -o mpp-sort-bitonic-1024 main.cu 

   nvcc  -DTHREADS_PER_BLOCK=512U  -arch sm_50 --allow-unsupported-compiler  -std=c++17 -Xcompiler=-std=c++17 -ccbin /usr/bin/g++-12 -o mpp-sort-bitonic-512 main.cu 

   nvcc  -DTHREADS_PER_BLOCK=256U  -arch sm_50 --allow-unsupported-compiler  -std=c++17 -Xcompiler=-std=c++17 -ccbin /usr/bin/g++-12 -o mpp-sort-bitonic-256 main.cu 
fi

   
if [ $host = "t101" ]; then
      echo "COMO compilar para maquina t101 ???"
fi




