#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "chrono.c"

__global__ void blockAndGlobalHisto(int **HH, int *Hg, int h, int *Input, int nElements, int minV, int maxV) {
    extern __shared__ int Histogram[];

    int partition_size = (maxV - minV)/h;
    Histogram[threadIdx.x] = 0;

    int partition;
    int total_active_threads = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < nElements; i += total_active_threads) {
        partition = Input[i] / partition_size;
        atomicAdd(Histogram[partition], 1);
        atomicAdd(Hg[partition], 1);
    }

    if(threadIdx.x == 0)
        cudaMemcpy(HH[blockIdx.x], Histogram, sizeof(int) * h, cudaMemcpyDeviceToDevice);
}
