#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "tcgen05.cuh"
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
  uint64_t clocks[8][32]; // [warp_idx][reg_idx] for profiling usage
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages, uint32_t kNumMulticast, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
    gemm_tcgen05_impl(__grid_constant__ const cute::TmaDescriptor a_desc,
                      __grid_constant__ const cute::TmaDescriptor b_desc,
                      __grid_constant__ const cute::TmaDescriptor cd_desc,
                      uint32_t m, uint32_t n, uint32_t k) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))

  using Allocator = cute::TMEM::Allocator1Sm;
  extern __shared__ __align__(1024) uint8_t __smem[];
  using SharedStorage = SharedStorage<BLOCK_M, BLOCK_N, BLOCK_K, kNumStages>;
  using Mma = MmaInstr<cutlass::bfloat16_t, cutlass::bfloat16_t, float, BLOCK_M,
                       BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  auto __sync_cluster = [&]() {
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();
  };
  auto tick = timestamp<true>;

  constexpr uint32_t kTmaAtomK = 128 / sizeof(cutlass::bfloat16_t);
  constexpr uint32_t kUmmaK = 32 / sizeof(cutlass::bfloat16_t);
  static_assert(BLOCK_K % kTmaAtomK == 0);
  static_assert(BLOCK_K % kUmmaK == 0);

  constexpr uint32_t kNumTmemCols = BLOCK_N * sizeof(cutlass::bfloat16_t) / 4;
  static_assert(Allocator::ColumnsPerAllocationSlice <= kNumTmemCols &&
                kNumTmemCols <= Allocator::Sm100TmemCapacityColumns);

  static_assert(kNumThreads % 128 == 0);
  static_assert(BLOCK_M == BLOCK_N,
                "Transposed TMEM epilogue demo assumes a square output tile");

  auto const smem = reinterpret_cast<SharedStorage *>(__smem);
  auto const warp_idx = cutlass::canonical_warp_idx_sync();
  auto const lane_idx = cutlass::canonical_lane_idx();
  auto const bidx = blockIdx.x;
  auto const bidy = blockIdx.y;
  // Prologue
  {
    if (warp_idx == 0) {
      if (cute::elect_one_sync()) {
        smem->clocks[7][31] = tick();
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
    if (cute::elect_one_sync()) {
      for (uint32_t kidx = 0; kidx < total_k_blocks; advance_pipeline(kidx)) {
        smem->tma_empty[stage].wait(phase ^ 1);
        smem->clocks[warp_idx][kidx] = tick();
        smem->tma_full[stage].arrive_and_expect_tx(
            (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(cutlass::bfloat16_t));
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t i = 0; i < BLOCK_K / kTmaAtomK; ++i) {
          cute::SM90_TMA_LOAD_2D::copy(
              &a_desc, reinterpret_cast<uint64_t *>(&smem->tma_full[stage]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem->a[stage] + i * BLOCK_M * kTmaAtomK,
              kidx * BLOCK_K + i * kTmaAtomK, bidx * BLOCK_M);
          cute::SM90_TMA_LOAD_2D::copy(
              &b_desc, reinterpret_cast<uint64_t *>(&smem->tma_full[stage]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem->b[stage] + i * BLOCK_N * kTmaAtomK,
              kidx * BLOCK_K + i * kTmaAtomK, bidy * BLOCK_N);
        }
      }
    }
  }
  // Umma
  else if (warp_idx == 1) {
    auto idesc = Mma::idesc();
    for (uint32_t kidx = 0; kidx < total_k_blocks; advance_pipeline(kidx)) {
      smem->tma_full[stage].wait(phase);
      smem->clocks[warp_idx][kidx] = tick();
      for (uint32_t k_inner = 0; k_inner < BLOCK_K / kUmmaK; ++k_inner) {
        uint32_t tile_k = k_inner * kUmmaK;
        auto desc_a = make_umma_smem_desc_swizzle128B<BLOCK_M, BLOCK_K>(
            smem->a[stage], 0, tile_k);
        auto desc_b = make_umma_smem_desc_swizzle128B<BLOCK_N, BLOCK_K>(
            smem->b[stage], 0, tile_k);
        Mma::fma(desc_b, desc_a, smem->tmem_ptr,
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
  }
  // Epilogue
  else {
    constexpr uint32_t kEpilogueWarps = 4;
    constexpr uint32_t kRowsPerTmemLoad = 32;
    static_assert(BLOCK_M % kRowsPerTmemLoad == 0);
    uint32_t __warp_idx = warp_idx % kEpilogueWarps;
    smem->umma_done.wait(0);
    smem->clocks[warp_idx][0] = tick();
    uint32_t n_idx = __warp_idx * 32 + lane_idx;
    cute::tma_store_fence();
    CUTLASS_PRAGMA_UNROLL
    for (uint32_t m_base = 0; m_base < BLOCK_M; m_base += kRowsPerTmemLoad) {
      uint32_t dregs[kRowsPerTmemLoad];
      tcgen05::utcld<32, 32, kRowsPerTmemLoad>(smem->tmem_ptr + m_base, dregs);
      cutlass::arch::fence_view_async_tmem_load();
      CUTLASS_PRAGMA_UNROLL
      for (uint32_t i = 0; i < kRowsPerTmemLoad; ++i) {
        smem->cd[(m_base + i) * BLOCK_N + n_idx] =
            cutlass::bfloat16_t(__uint_as_float(dregs[i]));
      }
      cutlass::arch::NamedBarrier::sync(128, 0);
      if (__warp_idx == 0 && cute::elect_one_sync()) {
        cute::SM90_TMA_STORE_2D::copy(&cd_desc, smem->cd + m_base * BLOCK_N,
                                      bidy * BLOCK_N, bidx * BLOCK_M + m_base);
      }
    }
    if (__warp_idx == 0 && cute::elect_one_sync()) {
      cute::tma_store_arrive();
      cute::tma_store_wait<0>();
    }
    smem->clocks[warp_idx][1] = tick();
  }
  __sync_cluster();
  if (warp_idx == 0) {
    Allocator().free(smem->tmem_ptr, kNumTmemCols);
  }
#if 0
  if (threadIdx.x == 0) {
    auto clocks = smem->clocks;
    BLOG("|S %lu\n", clocks[7][31]);
    for (uint32_t i = 0; i < total_k_blocks; ++i) {
      BLOG(" MMA %u: %lu\n", i, smem->clocks[1][i]);
    }
    BLOG(" Epilogue S: %lu\n", clocks[4][0]);
    BLOG("E|: %lu\n", clocks[4][1]);
  }
#endif

#else
  CUTE_INVALID_CONTROL_PATH("Error: TCGEN05 Required");
#endif
}
} // namespace arc
