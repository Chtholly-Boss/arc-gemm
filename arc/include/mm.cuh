#pragma once

// Thor Peak BF16 256FLOPS
// with 20 SMs -> 12.95 TFLOPS / SM

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "utils.cuh"

namespace arc {

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages>
struct SharedStorage {
  cutlass::bfloat16_t cd[BLOCK_M * BLOCK_N];
  cutlass::bfloat16_t a[kNumStages][BLOCK_M * BLOCK_K];
  cutlass::bfloat16_t b[kNumStages][BLOCK_N * BLOCK_K];
  cutlass::arch::ClusterTransactionBarrier tma_full[kNumStages];
  cutlass::arch::ClusterBarrier tma_empty[kNumStages];
  cutlass::arch::ClusterBarrier umma_done;
  alignas(16) uint32_t tmem_ptr;
  uint64_t clocks[8][16]; // [warp_idx][reg_idx] for profiling usage
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages, uint32_t kNumMulticast, uint32_t kNumThreads>
CUTLASS_GLOBAL void
gemm_tcgen05_impl(__grid_constant__ const cute::TmaDescriptor a_desc,
                  __grid_constant__ const cute::TmaDescriptor b_desc,
                  __grid_constant__ const cute::TmaDescriptor cd_desc,
                  uint32_t m, uint32_t n, uint32_t k) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))

  using Allocator = cute::TMEM::Allocator1Sm;
  extern __shared__ __align__(1024) uint8_t __smem[];
  using SharedStorage = SharedStorage<BLOCK_M, BLOCK_N, BLOCK_K, kNumStages>;
  using Mma = MmaInstr<cutlass::bfloat16_t, cutlass::bfloat16_t, float, 128,
                       128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  auto __sync_cluster = [&]() {
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();
  };

  constexpr uint32_t kTmaAtomK = 128 / sizeof(cutlass::bfloat16_t);
  constexpr uint32_t kUmmaK = 32 / sizeof(cutlass::bfloat16_t);
  static_assert(BLOCK_K % kTmaAtomK == 0);
  static_assert(BLOCK_K % kUmmaK == 0);

  constexpr uint32_t kNumTmemCols = BLOCK_N * sizeof(cutlass::bfloat16_t) / 4;
  static_assert(Allocator::ColumnsPerAllocationSlice <= kNumTmemCols &&
                kNumTmemCols <= Allocator::Sm100TmemCapacityColumns);

  constexpr uint32_t kNonEpilogueRegs = 40;
  constexpr uint32_t kEpilogueRegs = 232;
  static_assert(kNumThreads % 128 == 0);

  auto const smem = reinterpret_cast<SharedStorage *>(__smem);
  auto const warp_idx = cutlass::canonical_warp_idx_sync();
  auto const lane_idx = cutlass::canonical_lane_idx();

  // Prologue
  {
    if (warp_idx == 0) {
      if (cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&a_desc);
        cute::prefetch_tma_descriptor(&b_desc);
        cute::prefetch_tma_descriptor(&cd_desc);
      }
    } else if (warp_idx == 1) {
      if (cute::elect_one_sync()) {
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t i = 0; i < kNumStages; ++i) {
          smem->tma_full[i].init(1);
          smem->tma_empty[i].init(1);
        }
        smem->umma_done.init(1);
        cutlass::arch::fence_barrier_init();
      }
    } else if (warp_idx == 2) {
      Allocator().allocate(kNumTmemCols, &smem->tmem_ptr);
    }
  }
  __sync_cluster();

  uint32_t total_k_blocks = cutlass::ceil_div(k, BLOCK_K);
  uint32_t phase = 0;
  uint32_t stage = 0;
  auto advance_pipeline = [&](uint32_t &kidx) {
    ++kidx;
    stage = (stage + 1) % kNumStages;
    phase ^= (stage == 0);
  };

  // Tma Load
  if (warp_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<kNonEpilogueRegs>();
    if (cute::elect_one_sync()) {
      for (uint32_t kidx = 0; kidx < total_k_blocks; advance_pipeline(kidx)) {
        smem->tma_empty[stage].wait(phase ^ 1);
        smem->tma_full[stage].arrive_and_expect_tx(
            (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(cutlass::bfloat16_t));
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t i = 0; i < BLOCK_K / kTmaAtomK; ++i) {
          cute::SM90_TMA_LOAD_2D::copy(
              &a_desc, reinterpret_cast<uint64_t *>(&smem->tma_full[stage]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem->a[stage] + i * BLOCK_M * kTmaAtomK,
              kidx * BLOCK_K + i * kTmaAtomK, 0);
          cute::SM90_TMA_LOAD_2D::copy(
              &b_desc, reinterpret_cast<uint64_t *>(&smem->tma_full[stage]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem->b[stage] + i * BLOCK_N * kTmaAtomK,
              kidx * BLOCK_K + i * kTmaAtomK, 0);
        }
      }
    }
  }
  // Umma
  else if (warp_idx == 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNonEpilogueRegs>();
    auto idesc = Mma::idesc();
    for (uint32_t kidx = 0; kidx < total_k_blocks; advance_pipeline(kidx)) {
      smem->tma_full[stage].wait(phase);

      for (uint32_t k_inner = 0; k_inner < BLOCK_K / kUmmaK; ++k_inner) {
        uint32_t tile_k = k_inner * kUmmaK;
        auto desc_a = make_umma_smem_desc_swizzle128B<BLOCK_M, BLOCK_K>(
            smem->a[stage], 0, tile_k);
        auto desc_b = make_umma_smem_desc_swizzle128B<BLOCK_N, BLOCK_K>(
            smem->b[stage], 0, tile_k);
        Mma::fma(desc_a, desc_b, smem->tmem_ptr,
                 (kidx == 0 && k_inner == 0) ? 0u : 1u, idesc);
      }
      cutlass::arch::umma_arrive(
          reinterpret_cast<uint64_t const *>(&smem->tma_empty[stage]));
      __syncwarp();
    }
    cutlass::arch::umma_arrive(
        reinterpret_cast<uint64_t const *>(&smem->umma_done));
    __syncwarp();
  }
  // Pad to warpgroup size for register reconfiguration
  else if (warp_idx < 4) {
    cutlass::arch::warpgroup_reg_dealloc<kNonEpilogueRegs>();
  }
  // Epilogue
  else {
    constexpr uint32_t kEpilogueWarps = 4;
    using CopyOp = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x;
    cutlass::arch::warpgroup_reg_alloc<kEpilogueRegs>();

    uint32_t __warp_idx = warp_idx % kEpilogueWarps;
    smem->umma_done.wait(0);
    // TODO: Figure out the best granularity for loading from TMEM
    // 32dp32b<N>x with the optimal N to achieve peak throughput
    uint32_t row = __warp_idx * 32 + lane_idx;
    CUTLASS_PRAGMA_UNROLL
    for (uint32_t col = 0; col < BLOCK_N; col += 8) {
      CopyOp::DRegisters dregs;
      CopyOp::copy(smem->tmem_ptr + col, dregs[0], dregs[1], dregs[2], dregs[3],
                   dregs[4], dregs[5], dregs[6], dregs[7]);
      cutlass::arch::fence_view_async_tmem_load();
      CUTLASS_PRAGMA_UNROLL
      for (uint32_t i = 0; i < 8; ++i) {
        smem->cd[row * BLOCK_N + col + i] =
            cutlass::bfloat16_t(__uint_as_float(dregs[i]));
      }
    }
    cute::tma_store_fence();
    cutlass::arch::NamedBarrier::sync(128, 0);
    if (__warp_idx == 0 && cute::elect_one_sync()) {
      cute::SM90_TMA_STORE_2D::copy(&cd_desc, smem->cd, 0, 0);
      cute::tma_store_arrive();
      cute::tma_store_wait<0>();
    }
  }
  __sync_cluster();

  if (warp_idx == 0) {
    Allocator().free(smem->tmem_ptr, kNumTmemCols);
  }
#else
  CUTE_INVALID_CONTROL_PATH("Error: TCGEN05 Required");
#endif
}
} // namespace arc
