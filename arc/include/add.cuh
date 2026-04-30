#pragma once

#include <cstdint>

namespace arc {

template <typename T>
__global__ void vecadd_kernel(const T *a, const T *b, T *out, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

} // namespace arc
