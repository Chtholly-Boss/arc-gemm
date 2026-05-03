#pragma once

#include "common.cuh"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/util.hpp"
#include "cutlass/detail/helper_macros.hpp"

namespace arc {
inline void
make_2d_tma_desc(cute::TmaDescriptor *desc, void *ptr, uint64_t height,
                 uint64_t width, uint32_t box_height, uint32_t box_width,
                 CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {width, height};
  uint64_t globalStrides[rank - 1] = {width * sizeof(cutlass::bfloat16_t)};
  uint32_t boxDim[rank] = {box_width, box_height};
  uint32_t elementStrides[rank] = {1, 1};
  CHECK_CUDRV_ERROR(cuTensorMapEncodeTiled(
      desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

CUTLASS_DEVICE cute::UMMA::SmemDescriptor
make_umma_smem_desc(cute::UMMA::LayoutType layout_type, void *smem_ptr,
                    uint32_t sbo, uint32_t lbo) {
  cute::UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(layout_type);
  desc.start_address_ =
      static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(smem_ptr) >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = sbo >> 4;
  desc.leading_byte_offset_ = lbo >> 4;
  return desc;
}

// TODO : support more swizzle pattern
template <uint32_t BLOCK_MN, uint32_t BLOCK_K, typename dtype_t>
CUTLASS_DEVICE cute::UMMA::SmemDescriptor
make_umma_smem_desc_swizzle128B(dtype_t *base, uint32_t mn_idx,
                                uint32_t k_idx) {
  constexpr uint32_t kAtomBytes = 16;
  constexpr uint32_t kSwizzleBytes = 128;
  constexpr uint32_t kSwizzleK = kSwizzleBytes / sizeof(dtype_t);
  static_assert(BLOCK_K % kSwizzleK == 0);

  uint32_t k_slab = k_idx / kSwizzleK;
  uint32_t k_in_slab = k_idx % kSwizzleK;
  dtype_t *ptr =
      base + k_slab * BLOCK_MN * kSwizzleK + mn_idx * kSwizzleK + k_in_slab;
  return make_umma_smem_desc(cute::UMMA::LayoutType::SWIZZLE_128B, ptr,
                             8 * kSwizzleBytes, kAtomBytes);
}

template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major>
struct MmaInstr {
  using Op = cute::SM100_MMA_F16BF16_SS<a_type, b_type, c_type, M, N, a_major,
                                        b_major>;

  CUTE_HOST_DEVICE static constexpr uint64_t idesc() {
    return cute::UMMA::make_runtime_instr_desc<a_type, b_type, c_type, M, N,
                                               a_major, b_major>();
  }

  CUTLASS_DEVICE static void fma(uint64_t desc_a, uint64_t desc_b,
                                 uint32_t tmem_c, uint32_t scale_c,
                                 uint64_t runtime_instr_desc) {
    Op::fma(desc_a, desc_b, tmem_c, scale_c, runtime_instr_desc);
  }
};
} // namespace arc
