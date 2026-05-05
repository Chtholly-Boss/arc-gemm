#pragma once
#include "common.cuh"
#include "cute/arch/copy_sm90_desc.hpp"
#include <cstdint>

namespace arc::tma {
namespace detail {

template <uint32_t kSwizzleBytes, uint32_t kAtomBytes>
inline CUtensorMapSwizzle to_CUtensorMapSwizzle() {
  return kAtomBytes == 64       ? CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
         : kAtomBytes == 32     ? CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
         : kSwizzleBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
         : kSwizzleBytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B
         : kSwizzleBytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B
                                : CU_TENSOR_MAP_SWIZZLE_NONE;
}

template <typename T, uint32_t kSwizzleBytes, uint32_t kAtomBytes,
          uint32_t rank>
void make_tma_desc_impl(cute::TmaDescriptor *desc, void *ptr,
                        uint64_t const (&dims)[rank],
                        uint32_t const (&logicalBoxDim)[rank]) {
  auto swizzle = to_CUtensorMapSwizzle<kSwizzleBytes, kAtomBytes>();
  auto data_type = cute::TMA::to_CUtensorMapDataType<T>();

  uint32_t boxDim[rank];
  uint32_t elementStrides[rank];
  uint64_t globalDim[rank];
  for (uint32_t i = 0; i < rank; ++i) {
    auto tma_idx = rank - 1 - i;
    globalDim[i] = dims[tma_idx];
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
    // boxDim specifies elements traversed along each tensorRank dimension and
    // must be non-zero and less than or equal to 256.
    boxDim[i] = logicalBoxDim[tma_idx];
    elementStrides[i] = 1;
  }

  uint64_t globalStrides[rank > 1 ? rank - 1 : 1] = {};
  uint64_t stride = sizeof(T);
  for (uint32_t i = 0; i + 1 < rank; ++i) {
    stride *= globalDim[i];
    globalStrides[i] = stride;
  }

  CHECK_CUDRV_ERROR(cuTensorMapEncodeTiled(
      desc, data_type, rank, ptr, globalDim, globalStrides, boxDim,
      elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

} // namespace detail

template <typename T = uint8_t, uint32_t kSwizzleBytes = 0,
          uint32_t kAtomBytes = 16, uint32_t... boxDims>
void make_tma_desc(cute::TmaDescriptor *desc, void *ptr,
                   uint64_t const (&dims)[sizeof...(boxDims)]) {
  constexpr uint32_t rank = sizeof...(boxDims);
  static_assert(rank > 0, "make_tma_desc expects at least one box dimension");
  uint32_t const logicalBoxDim[rank] = {boxDims...};
  detail::make_tma_desc_impl<T, kSwizzleBytes, kAtomBytes>(desc, ptr, dims,
                                                           logicalBoxDim);
}

template <typename T = uint8_t, uint32_t kSwizzleBytes = 0,
          uint32_t kAtomBytes = 16, uint32_t rank>
void make_tma_desc(cute::TmaDescriptor *desc, void *ptr,
                   uint64_t const (&dims)[rank],
                   uint32_t const (&boxDims)[rank]) {
  static_assert(rank > 0, "make_tma_desc expects at least one box dimension");
  detail::make_tma_desc_impl<T, kSwizzleBytes, kAtomBytes>(desc, ptr, dims,
                                                           boxDims);
}

} // namespace arc::tma
