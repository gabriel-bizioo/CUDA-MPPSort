// CI1009 2sem25
// Esqueleto para o programa de teste segmented-sort-bitonic
// voce vai incluir o seu kernel aqui para TESTA-LO

// Aluno 1: Gabriel Pucci Bizio
// Aluno 2:

#include <cuda_runtime.h>
#include <vector>
//#include <random>   // usaremos um OUTRO gerador de numeros 
                      //  (para poder compilar na orval nao usaremos o random padrao)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>  // For std::max and std::is_sorted

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "my_randomizer.h"  // usaremos um OUTRO gerador de numeros 
                      //  (para poder compilar na orval nao usaremos o random padrao)


// Minimal CUDA error checking
#define checkCudaErrors(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

typedef unsigned int uint;

#define MAX_SHARED_UINT 8192U // 48KB de memória compartilhada

// testaremos 3 tamanhos de blocos de threads
//  mas esses valores serao definidos na linha de compilação !
//const uint THREADS_PER_BLOCK = 256U;  // Must be <=1024, power of 2 recommended
//const uint THREADS_PER_BLOCK = 512U;  // Must be <=1024, power of 2 recommended
//const uint THREADS_PER_BLOCK = 1024U;  // Must be <=1024, power of 2 recommended

inline __device__ void Comparator(
    uint &keyA,
    uint &keyB,
    uint dir
) {
    uint t;
    if ((keyA > keyB) == dir) {
        t = keyA;
        keyA = keyB;
        keyB = t;
    }
}

/*
 * The KERNEL for Bitonic Sort of segments 
 *   (blockBitonicSort)
 *      is based on the NVIDIA Samples code
 * which has a copyright, SO, we state the copyright
 *
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm

////////////////////////////////////////////////////////////////////////////////
// Monolithic Bitonic Sort kernel for segments fitting into shared memory (out-of-place)
////////////////////////////////////////////////////////////////////////////////

__global__ void blockBitonicSort(   //   bitonicSortShared
    uint *d_DstKey,
    uint *d_SrcKey,
    uint *d_Offsets,
    uint *d_Sizes,
    uint dir
) {
    extern __shared__ uint s_key[];  // espaço em SHM para buscar o segmento a ser ordenado

    uint seg_idx = blockIdx.x;
    uint offset = d_Offsets[seg_idx];
    uint arrayLength = d_Sizes[seg_idx];
    if (arrayLength == 0) return;

    // Compute padded size (next power of 2 >= arrayLength)
    uint padded_size = (arrayLength == 1) ? 1 : (1 << (32 - __clz(arrayLength - 1)));
    uint pad_value = dir ? UINT_MAX : 0;  // Pad with max for ascending, min for descending
    uint tid = threadIdx.x;

    // Load data into shared memory with padding (looped over threads)
    for(uint i = threadIdx.x; i < padded_size; i += blockDim.x) {
        if(i < arrayLength)
            s_key[i] = d_SrcKey[offset + i];
        else
            s_key[i] = pad_value;
    }

    uint num_pairs = padded_size / 2;
    uint pairs_per_thread = num_pairs / blockDim.x;
    pairs_per_thread += (num_pairs % blockDim.x) ? 1 : 0;
    uint thread_pairs = tid * pairs_per_thread;
    uint active_threads = min(blockDim.x, num_pairs);
    __syncthreads();
    // Bitonic sort on padded size
    for (uint size = 2; size < padded_size; size <<= 1) {
        // Bitonic merge
        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            if(stride >= 32 && active_threads > 32) __syncthreads();
            for(uint i = 0; i < pairs_per_thread; i++) {
                uint pid = thread_pairs + i;
                uint ddd = dir ^ ((pid & (size / 2)) != 0);
                uint pos = 2 * pid - (pid & (stride - 1));
                Comparator(s_key[pos], s_key[pos + stride], ddd);
            }
        }
    }

    // Final bitonic merge step with ddd = dir
    for(uint stride = padded_size / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        for(int i = 0; i < pairs_per_thread; i++) {
            uint pid = thread_pairs + i;
            uint pos = 2 * pid - (pid & (stride - 1));
            Comparator(s_key[pos], s_key[pos + stride], dir);
        }
    }

    __syncthreads();
    for(uint i = threadIdx.x; i < arrayLength; i += blockDim.x)
        d_DstKey[offset + i] = s_key[i];
}


////////////////////////////////////////////////////////////////////////////////
// Interface function (out-of-place)
////////////////////////////////////////////////////////////////////////////////
void segmentedBitonicSort(
    uint *d_DstKey,
    uint *d_SrcKey,
    uint *d_Offsets,
    uint *d_Sizes,
    uint num_segments,
    uint dir,
    uint max_padded
) {
    if (num_segments == 0) return;

    // Launch one block per segment
    dim3 grid(num_segments, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    size_t shared_bytes = max_padded * sizeof(uint);

    //bitonicSortShared (OBS: a linha abaixo configura a shared memory e dispara o kernel blockBitonicSort)
    blockBitonicSort<<<grid, block, shared_bytes>>>(d_DstKey, d_SrcKey, d_Offsets, d_Sizes, dir);

    checkCudaErrors(cudaGetLastError());
}


void usage() {
    std::cout << "Usage: segmented_sort -n <total_elements> -segRange <min_seg> <max_seg>" << std::endl;
}

int main(int argc, char **argv) {
    uint n = 0;
    uint min_seg = 0, max_seg = 0;
    bool have_n = false, have_seg = false;

    // Manual command-line parsing
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<uint>(atoi(argv[++i]));
            have_n = true;
        } else if (strcmp(argv[i], "-segRange") == 0 && i + 2 < argc) {
            min_seg = static_cast<uint>(atoi(argv[++i]));
            max_seg = static_cast<uint>(atoi(argv[++i]));
            have_seg = true;
        }
    }

    if (!have_n || !have_seg || n == 0 || min_seg > max_seg) {
        usage();
        return 1;
    }

    // Seed random number generator
    //std::mt19937 gen(static_cast<unsigned int>(time(nullptr)));
    //std::uniform_int_distribution<uint> seg_dist(min_seg, max_seg);
    
    // usaremos um OUTRO gerador de numeros 
       //  (para poder compilar na orval nao usaremos o random padrao nessa distribuicao)
    Xoroshiro128Plus gen(static_cast<uint64_t>(time(nullptr)));   // ← same as before
    // uniform_int_distribution works perfectly with any generator that has operator()
    std::uniform_int_distribution<uint> seg_dist(min_seg, max_seg);


    // Generate segment sizes
    std::vector<uint> sizes;
    uint total = 0;
    while (total < n) {
        uint seg = seg_dist(gen);
        if (total + seg > n) {
            seg = n - total;
        }
        if (seg > 0) {
            sizes.push_back(seg);
            total += seg;
        }
    }
    std::cout << "Number of segments Generated:" << sizes.size() << std::endl;
    // Compute offsets
    std::vector<uint> offsets(sizes.size());
    offsets[0] = 0;
    for (size_t i = 1; i < sizes.size(); ++i) {
        offsets[i] = offsets[i - 1] + sizes[i - 1];
    }

    // initialize (GENERATE) Input vector  h_src
    std::vector<uint> h_src(n);  // declara
    
    // unsigned int Input[MAX_SIZE];    // estava assim na especificacao
     // gera a entrada
     int inputSize = 0;
     for( int i = 0; i < n; i++ ){
     
        // aqui vamos usar o random sem problemas!
	int a = rand();  // Returns a pseudo-random integer
	                 //    between 0 and RAND_MAX.
	int b = rand();  // same as above
	
	unsigned int v = a * 100 + b;

        // inserir o valor v na posição i
	h_src[ i ] = (unsigned int) v;
     }
     inputSize = n;
     
    // Compute max padded_size
    uint max_padded = 0;
    for (uint size : sizes) {
        uint padded = (size == 0) ? 0 : (size == 1) ? 1 : (1u << (32 - __builtin_clz(size - 1))); // ????????????
        max_padded = std::max(max_padded, padded);
    }

    // Device memory allocation
    uint *d_src = nullptr, *d_dst = nullptr, *d_offsets = nullptr, *d_sizes = nullptr;
    uint *d_copy = nullptr, *d_inplace = nullptr;
    checkCudaErrors(cudaMalloc(&d_src, n * sizeof(uint)));
    checkCudaErrors(cudaMalloc(&d_dst, n * sizeof(uint)));
    checkCudaErrors(cudaMalloc(&d_offsets, sizes.size() * sizeof(uint)));
    checkCudaErrors(cudaMalloc(&d_sizes, sizes.size() * sizeof(uint)));
    checkCudaErrors(cudaMalloc(&d_copy, n * sizeof(uint)));
    checkCudaErrors(cudaMalloc(&d_inplace, n * sizeof(uint)));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_src, h_src.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_offsets, offsets.data(), sizes.size() * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sizes, sizes.data(), sizes.size() * sizeof(uint), cudaMemcpyHostToDevice));

    // Duplicate the input vector for Thrust
    checkCudaErrors(cudaMemcpy(d_copy, d_src, n * sizeof(uint), cudaMemcpyDeviceToDevice));

    // Duplicate for in-place bitonic
    checkCudaErrors(cudaMemcpy(d_inplace, d_src, n * sizeof(uint), cudaMemcpyDeviceToDevice));

    // Timing for out-of-place segmented bitonic sort
    cudaEvent_t start_bitonic_oop, stop_bitonic_oop;
    checkCudaErrors(cudaEventCreate(&start_bitonic_oop));
    checkCudaErrors(cudaEventCreate(&stop_bitonic_oop));

    checkCudaErrors(cudaEventRecord(start_bitonic_oop, 0));

    // Call out-of-place segmented sort (dir=1 for ascending)
    segmentedBitonicSort(d_dst, d_src, d_offsets, d_sizes,
            static_cast<uint>(sizes.size()), 1, max_padded);

    checkCudaErrors(cudaEventRecord(stop_bitonic_oop, 0));
    checkCudaErrors(cudaEventSynchronize(stop_bitonic_oop));

    float elapsed_ms_bitonic_oop = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsed_ms_bitonic_oop, start_bitonic_oop, stop_bitonic_oop));

    float time_sec_bitonic_oop = elapsed_ms_bitonic_oop / 1000.0f;
    float throughput_bitonic_oop = (time_sec_bitonic_oop > 0.0f) ? (static_cast<float>(n) / time_sec_bitonic_oop / 1e9f) : 0.0f;

    std::cout << "Segmented Bitonic Sort (Out-of-Place):" << std::endl;
    std::cout << "Time taken: " << time_sec_bitonic_oop << " seconds" << std::endl;
    std::cout << "Throughput: " << throughput_bitonic_oop << " GEls/s" << std::endl;

    // Timing for Thrust sort
    cudaEvent_t start_thrust, stop_thrust;
    checkCudaErrors(cudaEventCreate(&start_thrust));
    checkCudaErrors(cudaEventCreate(&stop_thrust));

    checkCudaErrors(cudaEventRecord(start_thrust, 0));

    // Sort the entire copy with Thrust sort (key-only)
    thrust::device_ptr<uint> thrust_begin(d_copy);
    thrust::device_ptr<uint> thrust_end = thrust_begin + n;
    thrust::sort(thrust_begin, thrust_end);

    checkCudaErrors(cudaEventRecord(stop_thrust, 0));
    checkCudaErrors(cudaEventSynchronize(stop_thrust));

    float elapsed_ms_thrust = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsed_ms_thrust, start_thrust, stop_thrust));

    float time_sec_thrust = elapsed_ms_thrust / 1000.0f;
    float throughput_thrust = (time_sec_thrust > 0.0f) ? (static_cast<float>(n) / time_sec_thrust / 1e9f) : 0.0f;

    std::cout << "Thrust Sort:" << std::endl;
    std::cout << "Time taken: " << time_sec_thrust << " seconds" << std::endl;
    std::cout << "Throughput: " << throughput_thrust << " GEls/s" << std::endl;

    // Speedup of out-of-place bitonic over thrust
    float speedup_oop_vs_thrust = (time_sec_thrust > 0.0f) ? (time_sec_thrust / time_sec_bitonic_oop) : 0.0f;
    std::cout << "Speedup (Thrust time / segmented-bitonic time): " << speedup_oop_vs_thrust << "x" << std::endl;

    // Verification for out-of-place bitonic
    std::vector<uint> h_dst(n);
    checkCudaErrors(cudaMemcpy(h_dst.data(), d_dst, n * sizeof(uint), cudaMemcpyDeviceToHost));

    bool all_sorted_oop = true;
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto begin = h_dst.begin() + offsets[i];
        auto end = begin + sizes[i];
        if (!std::is_sorted(begin, end)) {
            std::cerr << "Out-of-Place Segment " << i << " (offset " << offsets[i] << ", size " << sizes[i] << ") is not sorted!" << std::endl;
            all_sorted_oop = false;
        }
    }

    if (all_sorted_oop) {
        std::cout << "Verification (Bitonic Out-of-Place): All segments sorted correctly." << std::endl;
    } else {
        std::cout << "Verification (Bitonic Out-of-Place): Some segments are not sorted." << std::endl;
    }

    // Optional: Verify Thrust sort
    std::vector<uint> h_copy(n);
    checkCudaErrors(cudaMemcpy(h_copy.data(), d_copy, n * sizeof(uint), cudaMemcpyDeviceToHost));
    if (std::is_sorted(h_copy.begin(), h_copy.end())) {
        std::cout << "Verification (Thrust): Entire array sorted correctly." << std::endl;
    } else {
        std::cout << "Verification (Thrust): Entire array not sorted." << std::endl;
    }

    // Cleanup
    checkCudaErrors(cudaEventDestroy(start_bitonic_oop));
    checkCudaErrors(cudaEventDestroy(stop_bitonic_oop));
    checkCudaErrors(cudaEventDestroy(start_thrust));
    checkCudaErrors(cudaEventDestroy(stop_thrust));
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_dst));
    checkCudaErrors(cudaFree(d_offsets));
    checkCudaErrors(cudaFree(d_sizes));
    checkCudaErrors(cudaFree(d_copy));
    checkCudaErrors(cudaFree(d_inplace));

    return (all_sorted_oop ) ? 0 : 1;
}
