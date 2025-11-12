#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <thrust.h>

#include "chrono.c"
#define DEBUG

inline cudaError_t 
checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(__DEBUG)
    if(result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
}

__global__ void 
blockAndGlobalHistogram(unsigned int *HH, unsigned int *Hg, int h, unsigned int *Input,
        unsigned int nElements, unsigned int minV, unsigned int maxV) {
    extern __shared__ int histogram[];

    int partition_size = (maxV - minV)/h;
    histogram[threadIdx.x] = 0;

    int partition;
    int total_active_threads = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < nElements; i += total_active_threads) {
        partition = Input[i] / partition_size;
        atomicAdd(histogram[partition], 1);
    }

    if(threadIdx.x == 0) {
        for(int i = 0; i < h; i++)
            atomicAdd(Hg[i], histogram[i]);
    }

    // Escreve a linha do bloco na matriz global
    if(threadIdx.x == 0)
        checkCuda(cudaMemcpy(HH[blockIdx.x * h], histogram, sizeof(unsigned int) * h, cudaMemcpyDeviceToDevice));
}

__global__ void
globalHistogramScan(unsigned int *Hg, unsigned int *SHg, int h) {
    extern __shared__ int scan[];

    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x == 0) {
        scan[0] = 0;
        checkCuda(cudaMemcpy(scan[1], Hg, sizeof(unsigned int) * h, cudaMemcpyDeviceToDevice));
    }

    for(int i = 1; i <= h; i+=i) {
        if(idx <= h && idx - i >= 0)
            scan[idx] += scan[idx - i];
        __syncthreads();
    }

    if(threadIdx.x == 0)
        checkCuda(cudaMemcpy(SHg, scan, sizeof(unsigned int) * h, cudaMemcpyDeviceToDevice));
}

__global__ void
verticalScanHH(unsigned int *Hg, unsigned int *HH, unsigned int *PSv, int h, int nb) {
    // sizeof(int) * h * nb
    extern __shared__ int scan[];

    //blockDim.x = h
    PSv[0 + threadIdx.x] = 0;

    //blockDim.y = nb
    scan[threadIdx.y * nb + threadIdx.x] = HH[threadIdx.x * h + threadIdx.y];

    for(int i = 0; i < nb; i += i) {
        if(threadIdx.x <= h && threadIdx.x - i >= 0)
            scan[threadIdx.y * nb + threadIdx.x] += scan[threadIdx.y * nb + threadIdx.x - i];
        __syncthreads();
    }

    if(threadIdx.y < nb - 1)
        PSv[threadIdx.x * h + threadIdx.y+1] = scan[threadIdx.y * nb + threadIdx.x];
}

__global__ void
partitionKernel(unsigned int *HH, unsigned int *SHg, unsigned int *PSv, int h, unsigned int *Input,
        int nElements, unsigned int vMin, unsigned int vMax) {
    extern __shared__ int histogram[];

    if(threadIdx.x == 0)
        checkCuda(cudaMemcpy(histogram, PSv[blockIdx.x * h], sizeof(unsigned int) * h, cudaMemcpyDeviceToDevice));

    if(threadIdx.x < h)
        histogram[threadIdx.x] += SHg[threadIdx.x];


}
