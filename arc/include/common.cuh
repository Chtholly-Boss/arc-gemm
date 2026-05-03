#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHECK_CUDA_ERROR(err)                                                  \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",                \
              cudaGetErrorString(err_), err_, __FILE__, __LINE__);             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_CUDRV_ERROR(err)                                                 \
  do {                                                                         \
    CUresult err_ = (err);                                                     \
    if (err_ != CUDA_SUCCESS) {                                                \
      const char *err_str;                                                     \
      cuGetErrorString(err_, &err_str);                                        \
      fprintf(stderr, "CUDA Driver Error: %s (err_num=%d) at %s:%d\n",         \
              err_str, err_, __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <bool UseGlobalTimer> __device__ __forceinline__ uint64_t timestamp() {
  uint64_t t;
  if constexpr (UseGlobalTimer) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } else {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  }
  return t;
}

__device__ __forceinline__ uint32_t __smid() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

#define BLOG(fmt, ...)                                                         \
  printf("#P[%u]#B[%u,%u,%u] " fmt, __smid(), blockIdx.x, blockIdx.y,          \
         blockIdx.z, ##__VA_ARGS__)
#define TLOG(fmt, ...)                                                         \
  printf("#B[%u]T[%u] " fmt, blockIdx.x, threadIdx.x, ##__VA_ARGS__)
