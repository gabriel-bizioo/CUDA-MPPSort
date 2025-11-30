#include <climits>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define MAX_SIZE INT_MAX
#define SHM_LIMIT 4096


inline void checkCuda(cudaError_t result, const char debugstr[256]) {
    if(result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s (%s)\n", cudaGetErrorString(result), debugstr);
        assert(result == cudaSuccess);
    }
}

inline void printDeviceArray(unsigned int *d_Array, int nElements, const char debugstr[256]) {
#ifdef DEBUG
    unsigned int *h_Array = (unsigned int*) malloc(sizeof(unsigned int) * nElements);
    checkCuda(cudaMemcpy(h_Array, d_Array, sizeof(unsigned int) * nElements, cudaMemcpyDeviceToHost), debugstr);

    printf("Imprimindo array (%s):\n", debugstr);
    for(int i = 0; i < nElements; i++)
        printf("%u ", h_Array[i]);
    printf("\n\n");
    free(h_Array);
#endif
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

    if(threadIdx.x < h)
        histogram[threadIdx.x] = 0;
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
        HH[blockIdx.x * h + threadIdx.x] = histogram[threadIdx.x];
        atomicAdd(&Hg[threadIdx.x], histogram[threadIdx.x]);
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
            scan[threadIdx.x] += scan[threadIdx.x - i];
        __syncthreads();
    }

    if(threadIdx.x < h) {
        SHg[threadIdx.x] = scan[threadIdx.x];
    }
}

__global__ void 
verticalScanHH(unsigned int *HH, unsigned int *PSv, int h) {
    if (threadIdx.x < h) {
        unsigned int index = blockIdx.x * h + threadIdx.x;

        unsigned int sum = 0;
        for (int i = 0; i < blockIdx.x; i++) {
            int hhIndex = i * h + threadIdx.x;
            sum += HH[hhIndex];
        }

        PSv[index] = sum;
    }
}

__global__ void
partitionKernel(unsigned int *SHg, unsigned int *PSv, int h, unsigned int *Input,
                unsigned int *Output, int nElements, unsigned int minV, unsigned int maxV) {
    extern __shared__ unsigned int HLsh[];

    if(threadIdx.x < h)
        HLsh[threadIdx.x] = PSv[blockIdx.x * h + threadIdx.x] + SHg[threadIdx.x];
    __syncthreads();

    unsigned int partition_size = (maxV - minV + 1) / h;
    if(partition_size == 0) partition_size = 1;

    int total_active_threads = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < nElements; i += total_active_threads) {
        unsigned int e = Input[i];
        unsigned int f = (e - minV) / partition_size;
        if(f == h) f = h - 1;

        unsigned int p = atomicAdd(&HLsh[f], 1);
        Output[p] = e;
    }
}

__device__ __host__ __forceinline__
unsigned int nextPowerOf2_portable(unsigned int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__device__ __host__ __forceinline__
unsigned int ceilDiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void blockBitonicSort(
    uint *d_DstKey,
    uint *d_SrcKey,
    uint *d_Offsets,
    uint *d_Sizes,
    uint dir
) {
    // Shared memory storage for one or more short vectors
    extern __shared__ uint s_key[];

    //Define a particao na qual o bloco vai trabalhar
    uint seg_idx = blockIdx.x;
    uint offset = d_Offsets[seg_idx];
    uint arrayLength = d_Sizes[seg_idx];
    if (arrayLength == 0) return;

    // Compute padded size (next power of 2 >= arrayLength)
    uint padded_size = (arrayLength == 1) ? 1 : (1 << (32 - __clz(arrayLength - 1)));
    uint pad_value = dir ?  UINT_MAX : 0;
    uint tid = threadIdx.x;

    for(uint i = tid; i < padded_size; i += blockDim.x) {
        if(i < arrayLength)
            s_key[i] = d_SrcKey[offset + i];
        else
            s_key[i] = pad_value;
    }

    uint pairs_per_thread = (padded_size / 2) / blockDim.x;
    pairs_per_thread += ((padded_size / 2) % blockDim.x) ? 1 : 0;
    uint thread_pairs = tid * pairs_per_thread;
    __syncthreads();
    // Bitonic sort on padded size
    for (uint size = 2; size < padded_size; size <<= 1) {
        // Bitonic merge
        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(uint i = 0; i < pairs_per_thread; i++) {
                uint pid = thread_pairs + i;
                uint ddd = dir ^((pid & (size / 2)) != 0);
                uint pos = 2 * pid - (pid & (stride - 1));
                Comparator(s_key[pos], s_key[pos + stride], ddd);
            }
        }
    }

    // ddd == dir for the last bitonic merge step
    for (uint stride = padded_size/ 2; stride > 0; stride >>= 1) {
        __syncthreads();
        for(uint i = 0; i < pairs_per_thread; i++) {
            uint pid = thread_pairs + i;
            uint pos = 2 * pid - (pid & (stride - 1));
            Comparator(s_key[pos], s_key[pos + stride], dir);
        }
    }

    __syncthreads();
    for(uint i = threadIdx.x; i < arrayLength; i += blockDim.x)
        d_DstKey[offset + i] = s_key[i];
}

bool verifySort(unsigned int *thrustOutput, unsigned int *Output_d, int nElements) {
    bool result = true;
    for(int i = 0; i < nElements; i++) {
        if(thrustOutput[i] != Output_d[i]) {
            result = false;
            break;
        }
    }

    return result;
}

void buildPSvCPU(const unsigned int *HH, unsigned int *PSv,
                 int B, int h)
{
    // HH e PSv têm tamanho B*h, indexação row-major:
    // linha = bloco (b), coluna = partição (j)
    // idx(b,j) = b*h + j

    for (int j = 0; j < h; ++j) {          // fixa partição (coluna)
        unsigned int acc = 0;
        for (int b = 0; b < B; ++b) {      // varre blocos (linhas)
            int idx = b * h + j;
            PSv[idx] = acc;
            acc += HH[idx];
        }
    }
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
        #ifdef DEBUG
            unsigned int v = a % (101);
        #else
            int b = rand();
            unsigned int v = a * 100 + b;
        #endif

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

    int nb;
    if(h >= 16)
        nb = 16;
    else
        nb = h;

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

    int nt;
    if(h > 1024)
        nt = 1024;
    else
        nt = h;

    // Warmup
    checkCuda(cudaMemset(HH, 0, sizeof(unsigned int) * nb * h), "Line 307");
    checkCuda(cudaMemset(Hg, 0, sizeof(unsigned int) * h), "Line 308");
    blockAndGlobalHistogram<<<nb, nt, sizeof(unsigned int) * h>>>(HH, Hg, h, Input_d,
            nTotalElements, minV, maxV);
    cudaDeviceSynchronize();
    unsigned int thrustQueue[h];

    float k1_time, k2_time, k3_time, k4_time, k5_time, total_time;
    k1_time = k2_time = k3_time = k4_time = k5_time = 0.0f;
    cudaError_t err;
    for(int rep = 0; rep < nReps; rep++) {
        float k1_temp, k2_temp, k3_temp, k4_temp, k5_temp;
        checkCuda(cudaMemset(HH, 0, sizeof(unsigned int) * nb * h), "Line 325\n");
        checkCuda(cudaMemset(Hg, 0, sizeof(unsigned int) * h), "Line 326");
        checkCuda(cudaMemset(PSv, 0, sizeof(unsigned int) * nb * h), "line 327");

        cudaEventRecord(k1_start);
        blockAndGlobalHistogram<<<nb, 1024, sizeof(unsigned int) * h>>>(HH, Hg, h,
                Input_d, nTotalElements, minV, maxV);
        cudaEventRecord(k1_stop);
        #ifdef DEBUG
            if (err != cudaSuccess) {
                err = cudaDeviceSynchronize();
                printf("Kernel failed: %s (blockAndGlobalHistogram)\n",
                        cudaGetErrorString(err));
                return 1;
            }
        #else
            cudaEventSynchronize(k1_stop);
        #endif

        cudaEventRecord(k2_start);
        globalHistogramScan<<<1, h, sizeof(unsigned int) * (h + 1)>>>(Hg, SHg, h);
        cudaEventRecord(k2_stop);
        #ifdef DEBUG
            if (err != cudaSuccess) {
                err = cudaDeviceSynchronize();
                printf("Kernel failed: %s (globalHistogramScan)\n",
                        cudaGetErrorString(err));
                return 1;
            }
        #else
            cudaEventSynchronize(k2_stop);
        #endif

        cudaEventRecord(k3_start);
        verticalScanHH<<<nb, h>>>(HH, PSv, h);
        cudaEventRecord(k3_stop);
        #ifdef DEBUG
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Kernel failed: %s (verticalScanHH)\n",
                        cudaGetErrorString(err));
                return 1;
            }
        #else
            cudaEventSynchronize(k3_stop);
        #endif

        cudaEventRecord(k4_start);
        partitionKernel<<<nb, 1024, sizeof(unsigned int) * h>>>(SHg, PSv, h, Input_d, Output_d, nTotalElements, minV, maxV);
        cudaEventRecord(k4_stop);
        #ifdef DEBUG
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Kernel failed: %s (partitionKernel)\n",
                        cudaGetErrorString(err));
                return 1;
            }
        #else
            cudaEventSynchronize(k4_stop);
        #endif

        cudaEventRecord(k5_start);
        size_t shared_bytes = SHM_LIMIT * sizeof(uint);
        blockBitonicSort<<<h, THREADS_PER_BLOCK, shared_bytes>>>(Output_d, Output_d, SHg, Hg, 1);
        checkCuda(cudaMemcpy(thrustQueue, Hg, sizeof(unsigned int) * h, cudaMemcpyDeviceToHost), "Thrust queue");
        for(int i = 0; i < h; i++) {
            if(thrustQueue[i] > SHM_LIMIT)
                thrust::sort(thrust::device, Output_d, Output_d + thrustQueue[i]);
        }
        cudaEventRecord(k5_stop);
        #ifdef DEBUG
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Kernel failed: %s (bitonicSortShared)\n",
                        cudaGetErrorString(err));
                return 1;
            }
        #else
            cudaEventSynchronize(k5_stop);
        #endif

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
    printf("Kernel 5 (bitonicSort + merge):     %.3f ms\n", k5_time);
    printf("Total mppSort time:                 %.3f ms\n\n", total_time);

    float mppSort_throughput = (nTotalElements / 1e6) / (total_time / 1000.0f);
    printf("======== Performance Metrics ========\n");
    printf("mppSort Throughput: %.3f GElements/s\n", mppSort_throughput / 1000.0f);

    printf("\n======== Verification ========\n");
    bool correct = verifySort(Output_h, Temp_h, nTotalElements);
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");


    float thrust_throughput = (nTotalElements / 1e6) / (thrust_total/ 1000.0f);
    printf("Thrust sort time:   %.3f ms\n", thrust_total);
    printf("Thrust Throughput:  %.3f GElements/s\n", thrust_throughput / 1000.0f);

    float speedup = thrust_total / total_time;
    printf("\nSpeedup (mppSort vs Thrust): %.2fx \n", fabs(speedup));


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

    cudaEventDestroy(thrust_start);
    cudaEventDestroy(thrust_stop);

    return 0;
}
