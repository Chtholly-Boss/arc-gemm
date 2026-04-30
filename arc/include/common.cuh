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
