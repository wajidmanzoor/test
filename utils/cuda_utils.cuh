#pragma once

#include "../inc/common.h"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cooperative_groups.h>
#include <cuda.h>

__device__ unsigned int globalCounter;

// Macro to check for CUDA errors after kernel launches
#define CUDA_CHECK_ERROR(kernelName)                                           \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error in kernel %s, file %s at line %d: %s\n", kernelName,  \
             __FILE__, __LINE__, cudaGetErrorString(err));                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define chkerr(code)                                                           \
  {                                                                            \
    chkerr_impl((code), __FILE__, __LINE__);                                   \
  }

inline void chkerr_impl(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code)
              << " | File: " << file << " | Line: " << line << std::endl;
    exit(-1);
  }
}

// Inline function for createLevelDataOffset
inline void createLevelDataOffset(cliqueLevelDataPointer levelData,
                                  ui offsetPartitionSize, ui totalWarps) {
  // Use thrust::transform with device execution policy
  thrust::transform(
      thrust::device,                    // Execution policy for device
      thrust::make_counting_iterator(0), // Start iterator (int)
      thrust::make_counting_iterator(
          static_cast<int>(totalWarps)), // End iterator (int)
      levelData.temp + 1,                // Output iterator (ui*)
      [=] __device__(int i) -> ui { // Lambda with explicit return type (ui)
        int task_count = levelData.count[i + 1];
        return (task_count > 0)
                   ? levelData
                         .offsetPartition[i * offsetPartitionSize + task_count]
                   : 0;
      });

  // Perform inclusive scan on temp array
  thrust::inclusive_scan(thrust::device, levelData.temp,
                         levelData.temp + totalWarps + 1, levelData.temp);

  // Perform inclusive scan on count array
  thrust::inclusive_scan(thrust::device, levelData.count,
                         levelData.count + totalWarps + 1, levelData.count);
}
