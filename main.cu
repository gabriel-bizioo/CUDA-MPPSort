#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <thrust.h>

#include "chrono.c"
#define DEBUG
#define MAX_SIZE 99999


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
partitionKernel(unsigned int *HH, unsigned int *SHg, unsigned int *PSv, int h, unsigned int *Input, unsigned int * Output,
        int nElements, unsigned int minV, unsigned int maxV) {
    extern __shared__ int histogram[];

    if(threadIdx.x == 0)
        checkCuda(cudaMemcpy(histogram, PSv[blockIdx.x * h], sizeof(unsigned int) * h, cudaMemcpyDeviceToDevice));

    int partition_size = (maxV - minV)/h;
    if(threadIdx.x < h)
        histogram[threadIdx.x] += SHg[threadIdx.x];

    int e, f, p;
    int total_active_threads = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < nElements; i += total_active_threads) {
        e = Input[i];
        f = Input[i] / partition_size;
        p = histogram[f];
        Output[p] = e;
    }

}

int main(int argc, char *argv[]) {

    unsigned int Input[MAX_SIZE];
    unsigned int nTotalElements;
    unsigned int inputSize = 0;
    int h;
    int nReps;

    nTotalElements = atoi(argv[1]);
    h = atoi(argv[2]);
    nReps = atoi(argv[3]);

    cudaEvent_t k1_start, k1_stop;
    cudaEvent_t k2_start, k2_stop;
    cudaEvent_t k3_start, k3_stop;
    cudaEvent_t k4_start, k4_stop;

    cudaEventCreate(&k1_start);
    cudaEventCreate(&k1_stop);
    cudaEventCreate(&k2_start);
    cudaEventCreate(&k2_stop);
    cudaEventCreate(&k3_start);
    cudaEventCreate(&k3_stop);
    cudaEventCreate(&k4_start);
    cudaEventCreate(&k4_stop);

    unsigned int *nElements_d;
    unsigned int *Input_d;
    unsigned int *Output_d;
    int *h_d;
    unsigned int *HH;
    unsigned int *Hg;
    unsigned int *SHg;
    unsigned int *PSv;
    int nb = 2 * 8;
    int nt = 1024;
    for(int i = 0; i < nReps; i++) {
        for( int j = 0; j < nTotalElements; j++ ) {
            int a = rand();     // Returns a pseudo-random integer
                                //    between 0 and RAND_MAX.
            int b = rand();     // same as above

            unsigned int v = a * 100 + b;

            // inserir o valor v na posição i
            Input[i] = (unsigned int) v;
        }
        inputSize = nTotalElements;
        
        checkCuda(cudaMalloc(Input_d, sizeof(unsigned int) * inputSize);
        checkCuda(cudaMemcpy(Input_d, &Input, nTotalElements * sizeof(unsigned int), cudaMemcpyHostToDevice));

        checkCuda(cudaMalloc(Output_d, sizeof(unsigned int) * inputSize));

        checkCuda(cudaMalloc(h_d, sizeof(int));
        checkCuda(cudaMemcpy(h_d, &h, sizeof(int), cudaMemcpyHostToDevice));

        checkCuda(cudaMalloc(HH, sizeof(unsigned int) * nb * h));
        checkCuda(cudaMemset(HH, 0, sizeof(unsigned int) * nb * h));

        checkCuda(cudaMalloc(Hg, sizeof(unsigned int) * h));
        checkCuda(cudaMemset(Hg, 0, sizeof(unsigned int) * h));

        checkCuda(cudaMalloc(SHg, sizeof(unsigned int) * (h + 1)));

        checkCuda(cudaMalloc(PSv, sizeof(unsigned int) * nb * h));
        checkCuda(cudaMemset(PSv, 0, sizeof(unsigned int) * nb * h));

        checkCuda(cudaMalloc(nElements_d, sizeof(unsigned int)));


        cudaEventRecord(k1_start);
        blockAndGlobalHistogram<<nb, nt, sizeof(unsigned int) * h>>(HH, Hg, h, Input_d, nElements, minV, maxV);
        cudaEventRecord(k1_stop);

        cudaEventRecord(k2_stop);
        globalHistogramScan<<1, nt, sizeof(unsigned int) * (h + 1)>>(Hg, SHg, h);
        cudaEventRecord(k2_stop);

        cudaEventRecord(k3_stop);
        // ajustar setting declarations
        verticalScanHH<<nb, nt>>(*Hg, *HH, *PSv, h_d, nb);
        cudaEventRecord(k3_stop);
        
        cudaEventRecord(k4_stop);
        partitionKernel(HH, SHg, PSv, h, Input, nElements, vMin, vMax);
        cudaEventRecord(k4_stop);

    }
}
