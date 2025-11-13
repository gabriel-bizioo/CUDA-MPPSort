#include <cstdlib>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define MAX_SIZE INT_MAX
#define SHARED_SIZE_LIMIT 1024

#define CUDA_ASSERT(cond, msg) \
    if (!(cond)) { \
        printf("[CUDA ASSERT FAILED] %s | block=(%d,%d,%d) thread=(%d,%d,%d)\n", \
               msg, blockIdx.x, blockIdx.y, blockIdx.z, \
               threadIdx.x, threadIdx.y, threadIdx.z); \
        asm("trap;"); \
    }


inline cudaError_t checkCuda(cudaError_t result, const char *debug) {
    if(result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s (%s)\n", cudaGetErrorString(result), debug);
        assert(result == cudaSuccess);
    }
    return result;
}

__device__ inline void Comparator(unsigned int &keyA, unsigned int &keyB, unsigned int dir) {
    unsigned int t;
    if ((keyA > keyB) == dir) {
        t = keyA;
        keyA = keyB;
        keyB = t;
    }
}

__global__ void
blockAndGlobalHistogram(unsigned int *HH, unsigned int *Hg, int h, unsigned int *Input,
        unsigned int nElements, unsigned int minV, unsigned int maxV) {
    extern __shared__ unsigned int histogram[];

    if(threadIdx.x < h) {
        histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    unsigned int partition_size = (maxV - minV + 1) / h;
    if(partition_size == 0) partition_size = 1;
    
    int total_active_threads = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < nElements; i += total_active_threads) {
        unsigned int val = Input[i];
        int partition = (val - minV) / partition_size;
        if(partition >= h) partition = h - 1;
        atomicAdd(&histogram[partition], 1);
    }
    __syncthreads();

    if(threadIdx.x < h) {
        atomicAdd(&Hg[threadIdx.x], histogram[threadIdx.x]);
        HH[blockIdx.x * h + threadIdx.x] = histogram[threadIdx.x];
    }
}

__global__ void
globalHistogramScan(unsigned int *Hg, unsigned int *SHg, int h) {
    // sizeof(unsigned int) * (h + 1)
    extern __shared__ unsigned int scan[];
 
    if(threadIdx.x == 0)
        scan[0] = 0;
    if(threadIdx.x < h)
        scan[threadIdx.x + 1] = Hg[threadIdx.x];
    __syncthreads();

    for(int i = 1; i <= h; i += i) {
        if(threadIdx.x <= h && threadIdx.x >= i)
            scan[threadIdx.x] = scan[threadIdx.x - i];
        __syncthreads();
    }

    if(threadIdx.x < h) {
        SHg[threadIdx.x] = scan[threadIdx.x];
    }
}

__global__ void verticalScanHH(unsigned int *HH, unsigned int *PSv, int h) {
    if (threadIdx.x < h) {
        int index = blockIdx.x * h + threadIdx.x;

        unsigned int sum = 0;
        for (int i = 0; i < blockIdx.x; i++) {
            int hhIndex = i * h + threadIdx.x;
            sum += HH[hhIndex];
        }
        PSv[index] = sum;
    }
}

__global__ void
partitionKernel(unsigned int *HH, unsigned int *SHg, unsigned int *PSv, int h, 
                unsigned int *Input, unsigned int *Output,
                int nElements, unsigned int minV, unsigned int maxV) {
    extern __shared__ unsigned int HLsh[];

    if(threadIdx.x < h) {
        HLsh[threadIdx.x] = PSv[blockIdx.x * h + threadIdx.x] + SHg[threadIdx.x];
    }
    __syncthreads();

    unsigned int partition_size = (maxV - minV + 1) / h;
    if(partition_size == 0) partition_size = 1;

    int total_active_threads = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < nElements; i += total_active_threads) {
        unsigned int e = Input[i];
        int f = (e - minV) / partition_size;
        if(f >= h) f = h - 1;

        unsigned int p = atomicAdd(&HLsh[f], 1);
        Output[p] = e;
    }
}

__global__ void
partitionBitonicSortShared(unsigned int *Output, const unsigned int *SHg, const unsigned int *Hg, int h, unsigned int dir) {
    __shared__ unsigned int s_key[SHARED_SIZE_LIMIT];

    for (int partition = blockIdx.x; partition < h; partition += gridDim.x) {
        unsigned int partition_start = SHg[partition];
        unsigned int partition_size  = Hg[partition];

        if (partition_size == 0) continue;

        for (unsigned int offset = 0; offset < partition_size; offset += SHARED_SIZE_LIMIT) {
            unsigned int tile_size = min(SHARED_SIZE_LIMIT, partition_size - offset);

            if (threadIdx.x < tile_size)
                s_key[threadIdx.x] = Output[partition_start + offset + threadIdx.x];
            __syncthreads();

            for (unsigned int size = 2; size <= tile_size; size <<= 1) {
                unsigned int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

                for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
                    __syncthreads();
                    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                    if (pos + stride < tile_size)
                        Comparator(s_key[pos], s_key[pos + stride], ddd);
                }
                __syncthreads();
            }

            if (threadIdx.x < tile_size)
                Output[partition_start + offset + threadIdx.x] = s_key[threadIdx.x];
        }
    }
}

__global__ void
partitionBitonicMergeGlobal(unsigned int *Output,   const unsigned int *SHg, const unsigned int *Hg, int h, unsigned int dir) {
    // Each block processes one or more partitions (stride by gridDim.x)
    for (int partition = blockIdx.x; partition < h; partition += gridDim.x) {

        unsigned int partition_start = SHg[partition];
        unsigned int partition_size  = Hg[partition];
        if (partition_size <= 1) continue;

        // Perform a global bitonic merge within the partition
        // This progressively merges sorted tiles into one fully sorted sequence.
        for (unsigned int size = 2; size <= partition_size; size <<= 1) {
            unsigned int ddd = dir ^ ((partition_start / size) & 1);

            for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
                unsigned int idx = partition_start + threadIdx.x + blockIdx.y * blockDim.x;
                if (idx >= partition_start + partition_size) continue;

                unsigned int ixj = idx ^ stride;  // bitonic partner
                if (ixj < partition_start || ixj >= partition_start + partition_size) continue;

                // Decide ascending or descending based on size block
                bool ascending = ((idx & size) == 0) ^ ddd;

                unsigned int a = Output[idx];
                unsigned int b = Output[ixj];
                if ((a > b) == ascending) {
                    Output[idx] = b;
                    Output[ixj] = a;
                }
            }
            __syncthreads(); // ensure all comparisons complete before next stride
        }
    }
}

bool verifySort(unsigned int *thrustOutput, unsigned int *Output_d, int nElements) {
    bool result = true;
    for(int i = 0; i < nElements; i++) {
        if(thrustOutput[i] != Output_d[i]) {
            result = false;
            printf("Mismatch at position %d: %u != %u\n", i, thrustOutput[i], Output_d[i]);
            break;
        }
    }

    return result;
}

int main(int argc, char *argv[]) {
    if(argc < 4) {
        printf("usage: ./mppSort <nTotalElements> h nR\n");
        printf("  nTotalElements: number of unsigned ints\n");
        printf("  h: number of histogram bins\n");
        printf("  nR: number of repetitions\n");
        return 1;
    }

    unsigned int nTotalElements = atoi(argv[1]);
    unsigned int h = atoi(argv[2]);
    int nReps = atoi(argv[3]);

    printf("======== mppSort Configuration ========\n");
    printf("Elements: %u\n", nTotalElements);
    printf("Histogram blocks: %d\n", h);
    printf("Repetitions: %d\n\n", nReps);

    unsigned int minV = UINT_MAX;
    unsigned int maxV = 0;

    unsigned int *Input = (unsigned int*)malloc(sizeof(unsigned int) * nTotalElements);

    srand(42);
    for(unsigned int i = 0; i < nTotalElements; i++) {
        int a = rand();
        int b = rand();
        unsigned int v = a * 100 + b;

        if(v < minV) minV = v;
        if(v > maxV) maxV = v;

        Input[i] = v;
    }

    unsigned int partition_width = (maxV - minV + 1) / h;
    printf("======== Input Statistics ========\n");
    printf("Interval: [%u, %u]\n", minV, maxV);
    printf("Partition width: %u\n\n", partition_width);

    unsigned int *Input_d, *Output_d;
    unsigned int *HH, *Hg, *SHg, *PSv;

    int nb = 16;
    int nt = 1024;

    checkCuda(cudaMalloc(&Input_d, sizeof(unsigned int) * nTotalElements), "Line 277\n");
    checkCuda(cudaMalloc(&Output_d, sizeof(unsigned int) * nTotalElements), "line 278\n");
    checkCuda(cudaMalloc(&HH, sizeof(unsigned int) * nb * h), "Line 279\n");
    checkCuda(cudaMalloc(&Hg, sizeof(unsigned int) * h), "Line 280\n");
    checkCuda(cudaMalloc(&SHg, sizeof(unsigned int) * (h + 1)), "Line 281\n");
    checkCuda(cudaMalloc(&PSv, sizeof(unsigned int) * nb * h), "Line 282\n");

    checkCuda(cudaMemcpy(Input_d, Input, sizeof(unsigned int) * nTotalElements, cudaMemcpyHostToDevice), "Line 284\n");

    cudaEvent_t k1_start, k1_stop;
    cudaEvent_t k2_start, k2_stop;
    cudaEvent_t k3_start, k3_stop;
    cudaEvent_t k4_start, k4_stop;
    cudaEvent_t k5_start, k5_stop;
    cudaEvent_t total_start, total_stop;

    cudaEventCreate(&k1_start);
    cudaEventCreate(&k1_stop);
    cudaEventCreate(&k2_start);
    cudaEventCreate(&k2_stop);
    cudaEventCreate(&k3_start);
    cudaEventCreate(&k3_stop);
    cudaEventCreate(&k4_start);
    cudaEventCreate(&k4_stop);
    cudaEventCreate(&k5_start);
    cudaEventCreate(&k5_stop);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    // Warmup
    checkCuda(cudaMemset(HH, 0, sizeof(unsigned int) * nb * h), "Line 307");
    checkCuda(cudaMemset(Hg, 0, sizeof(unsigned int) * h), "Line 308");
    blockAndGlobalHistogram<<<nb, nt, sizeof(unsigned int) * h>>>(HH, Hg, h, Input_d, nTotalElements, minV, maxV);
    cudaDeviceSynchronize();

    int maxBlocks = 1024;
    int numBlocks;
    if(h < maxBlocks)
        numBlocks = h;
    else
        numBlocks = maxBlocks;

    float k1_time, k2_time, k3_time, k4_time, k5_time, total_time;
    k1_time = k2_time = k3_time = k4_time = k5_time = 0.0f;
    cudaError_t err;
    for(int rep = 0; rep < nReps; rep++) {
        float k1_temp, k2_temp, k3_temp, k4_temp, k5_temp;
        checkCuda(cudaMemset(HH, 0, sizeof(unsigned int) * nb * h), "Line 325\n");
        checkCuda(cudaMemset(Hg, 0, sizeof(unsigned int) * h), "Line 326");
        checkCuda(cudaMemset(PSv, 0, sizeof(unsigned int) * nb * h), "line 327");

        cudaEventRecord(k1_start);
        blockAndGlobalHistogram<<<nb, nt, sizeof(unsigned int) * h>>>(HH, Hg, h, Input_d, nTotalElements, minV, maxV);
        if (err != cudaSuccess) {
            err = cudaDeviceSynchronize();
            printf("Kernel failed: %s (blockAndGlobalHistogram)\n", cudaGetErrorString(err));
            return 1;
        }
        cudaEventRecord(k1_stop);

        cudaEventRecord(k2_start);
        globalHistogramScan<<<1, nt, sizeof(unsigned int) * (h + 1)>>>(Hg, SHg, h);
        if (err != cudaSuccess) {
            err = cudaDeviceSynchronize();
            printf("Kernel failed: %s (globalHistogramScan)\n", cudaGetErrorString(err));
            return 1;
        }
        cudaEventRecord(k2_stop);

        cudaEventRecord(k3_start);
        verticalScanHH<<<nb, nt>>>(HH, PSv, h);
        cudaEventRecord(k3_stop);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel failed: %s (verticalScanHH)\n", cudaGetErrorString(err));
            return 1;
        }


        cudaEventRecord(k4_start);
        partitionKernel<<<nb, nt, sizeof(unsigned int) * h>>>(HH, SHg, PSv, h, Input_d, Output_d, nTotalElements, minV, maxV);
        cudaEventRecord(k4_stop);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel failed: %s (partitionKernel)\n", cudaGetErrorString(err));
            return 1;
        }

        cudaEventRecord(k5_start);
        partitionBitonicSortShared<<<numBlocks, nt/4, sizeof(unsigned int) * SHARED_SIZE_LIMIT>>>(Output_d, SHg, Hg, h, 0);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel failed: %s (partitionBitonicSortShared)\n", cudaGetErrorString(err));
            return 1;
        }
        partitionBitonicMergeGlobal<<<numBlocks, nt/4>>>(Output_d, SHg, Hg, h, 0);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel failed: %s (partitionBitonicMergeGlobal)\n", cudaGetErrorString(err));
            return 1;
        }
        cudaEventRecord(k5_stop);


        cudaEventElapsedTime(&k1_temp, k1_start, k1_stop);
        cudaEventElapsedTime(&k2_temp, k2_start, k2_stop);
        cudaEventElapsedTime(&k3_temp, k3_start, k3_stop);
        cudaEventElapsedTime(&k4_temp, k4_start, k4_stop);
        cudaEventElapsedTime(&k5_temp, k5_start, k5_stop);

        k1_time += k1_temp;
        k2_time += k2_temp;
        k3_time += k3_temp;
        k4_time += k4_temp;
        k5_time += k5_temp;
    }

    k1_time /= nReps;
    k2_time /= nReps;
    k3_time /= nReps;
    k4_time /= nReps;
    k5_time /= nReps;
    total_time = k1_time + k2_time + k3_time + k4_time + k5_time;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    unsigned int *Temp_d;
    checkCuda(cudaMalloc(&Temp_d, sizeof(unsigned int) * nTotalElements), "Line 374\n");

    cudaEvent_t thrust_start, thrust_stop;
    cudaEventCreate(&thrust_start);
    cudaEventCreate(&thrust_stop);

    float thrust_total = 0.0f;
    for(int rep = 0; rep < nReps; rep++) {
        checkCuda(cudaMemcpy(Temp_d, Input_d, sizeof(unsigned int) * nTotalElements, cudaMemcpyDeviceToDevice), "Thrust cudaMemcpy");

        float thrust_temp = 0.0f;
        thrust::device_ptr<unsigned int> dev_ptr(Temp_d);
        cudaEventRecord(thrust_start);
        thrust::sort(thrust::device, dev_ptr, dev_ptr + nTotalElements);
        cudaEventRecord(thrust_stop);
        cudaEventSynchronize(thrust_stop);
        cudaEventElapsedTime(&thrust_temp, thrust_start, thrust_stop);

        thrust_total += thrust_temp;
    }
    unsigned int *Temp_h = (unsigned int *)malloc(sizeof(unsigned int) * nTotalElements);
    unsigned int *Output_h = (unsigned int *) malloc(sizeof(unsigned int ) * nTotalElements);
    checkCuda(cudaMemcpy(Temp_h, Temp_d, sizeof(unsigned int) * nTotalElements, cudaMemcpyDeviceToHost), "Line 400");
    checkCuda(cudaMemcpy(Output_h, Output_d, sizeof(unsigned int) * nTotalElements, cudaMemcpyDeviceToHost), "line 461");

    thrust_total /= nReps;


    printf("======== Kernel Timing (Average over %d runs) ========\n", nReps);
    printf("Kernel 1 (blockAndGlobalHistogram): %.3f ms\n", k1_time);
    printf("Kernel 2 (globalHistogramScan):     %.3f ms\n", k2_time);
    printf("Kernel 3 (verticalScanHH):          %.3f ms\n", k3_time);
    printf("Kernel 4 (partitionKernel):         %.3f ms\n", k4_time);
    printf("Kernel 5 (bitonicSort + Thrust):    %.3f ms\n", k5_time);
    printf("Total mppSort time:                 %.3f ms\n\n", total_time);

    float mppSort_throughput = (nTotalElements / 1e6) / (total_time / 1000.0f);
    printf("======== Performance Metrics ========\n");
    printf("mppSort Throughput: %.3f GElements/s\n", mppSort_throughput / 1000.0f);

    printf("\n======== Verification ========\n");
    bool correct = verifySort(Output_h, Temp_h, nTotalElements);
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("Print 4\n");


    float thrust_throughput = (nTotalElements / 1e6) / (thrust_total/ 1000.0f);
    printf("Thrust sort time:   %.3f ms\n", thrust_total);
    printf("Thrust Throughput:  %.3f GElements/s\n", thrust_throughput / 1000.0f);

    float speedup = thrust_total / total_time;
    printf("\nSpeedup (mppSort vs Thrust): %.2fx n", fabs(speedup));


    free(Input);
    cudaFree(Input_d);
    cudaFree(Output_d);
    cudaFree(HH);
    cudaFree(Hg);
    cudaFree(SHg);
    cudaFree(PSv);
    cudaFree(Temp_d);

    cudaEventDestroy(k1_start);
    cudaEventDestroy(k1_stop);
    cudaEventDestroy(k2_start);
    cudaEventDestroy(k2_stop);
    cudaEventDestroy(k3_start);
    cudaEventDestroy(k3_stop);
    cudaEventDestroy(k4_start);
    cudaEventDestroy(k4_stop);
    cudaEventDestroy(k5_start);
    cudaEventDestroy(k5_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaEventDestroy(thrust_start);
    cudaEventDestroy(thrust_stop);

    return 0;
}
