#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/numeric/math.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "tcgen05.cuh"
#include "utils.cuh"

namespace arc {

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t kNumStages,
          typename a_type, typename b_type, typename cd_type>
struct SharedStorage {
  cd_type cd[BM * BN];
  a_type a[kNumStages][BM * BK];
  b_type b[kNumStages][BN * BK];
  cutlass::arch::ClusterTransactionBarrier tma_full[kNumStages];
  cutlass::arch::ClusterBarrier tma_empty[kNumStages];
  cutlass::arch::ClusterBarrier umma_done;
  alignas(16) uint32_t tmem_ptr;
  uint64_t clocks[8][32];
};

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t kNumStages,
          uint32_t BPC, uint32_t TPB>
CUTLASS_GLOBAL void __cluster_dims__(BPC, 1, 1) __launch_bounds__(TPB)
    gemm_tcgen05_impl(__grid_constant__ const cute::TmaDescriptor a_desc,
                      __grid_constant__ const cute::TmaDescriptor b_desc,
                      __grid_constant__ const cute::TmaDescriptor cd_desc,
                      uint32_t m, uint32_t n, uint32_t k) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
  static_assert(TPB % 128 == 0);
  static_assert(BN == 128);
  static_assert(BM % 16 == 0);

  using a_type = cutlass::bfloat16_t;
  using b_type = cutlass::bfloat16_t;
  using c_type = cutlass::bfloat16_t;
  using acc_type = float;
  auto constexpr a_major = cute::UMMA::Major::K;
  auto constexpr b_major = cute::UMMA::Major::K;

  using Mma = MmaInstr<a_type, b_type, acc_type, BN, BM, a_major, b_major>;
  constexpr uint32_t kUmmaK = 32 / sizeof(a_type);
  static_assert(BK % kUmmaK == 0);

  extern __shared__ __align__(1024) uint8_t __smem[];
  using SharedStorage =
      SharedStorage<BM, BN, BK, kNumStages, a_type, b_type, c_type>;
  auto const smem = reinterpret_cast<SharedStorage *>(__smem);

  using Allocator = cute::TMEM::Allocator1Sm;
  constexpr uint32_t kNumTmemCols = cute::bit_ceil(BM);
  static_assert(Allocator::ColumnsPerAllocationSlice <= kNumTmemCols &&
                kNumTmemCols <= Allocator::Sm100TmemCapacityColumns);

  auto __sync_cluster = [&]() {
    BPC > 1 ? cute::cluster_sync() : __syncthreads();
  };
  auto probe = timestamp<false>;

  auto const warp_idx = cutlass::canonical_warp_idx_sync();
  auto const lane_idx = cutlass::canonical_lane_idx();
  auto const bidx = blockIdx.x;
  auto const bidy = blockIdx.y;
  // Prologue
  {
    if (warp_idx == 0) {
      if (cute::elect_one_sync()) {
        probe(smem->clocks[7][31]);
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

  uint32_t total_k_blocks = cutlass::ceil_div(k, BK);
  uint32_t phase = 0;
  uint32_t stage = 0;
  auto advance_pipeline = [&](uint32_t &kidx) {
    ++kidx;
    stage = (stage + 1) % kNumStages;
    phase ^= (stage == 0); // loop back to stage 0, flip phase
  };

  // Tma Load
  if (warp_idx == 0) {
    if (cute::elect_one_sync()) {
      for (uint32_t kidx = 0; kidx < total_k_blocks; advance_pipeline(kidx)) {
        smem->tma_empty[stage].wait(phase ^ 1);
        probe(smem->clocks[warp_idx][kidx]);
        smem->tma_full[stage].arrive_and_expect_tx(BM * BK * sizeof(a_type) +
                                                   BN * BK * sizeof(b_type));
        constexpr uint32_t TmaAtomA = 128 / sizeof(a_type);
        static_assert(BK % TmaAtomA == 0);
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t i = 0; i < BK / TmaAtomA; ++i) {
          cute::SM90_TMA_LOAD::copy(
              &a_desc, reinterpret_cast<uint64_t *>(&smem->tma_full[stage]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem->a[stage] + i * BM * TmaAtomA, kidx * BK + i * TmaAtomA,
              bidx * BM);
        }
        constexpr uint32_t TmaAtomB = 128 / sizeof(b_type);
        static_assert(BK % TmaAtomB == 0);
        for (uint32_t i = 0; i < BK / TmaAtomB; ++i) {
          cute::SM90_TMA_LOAD::copy(
              &b_desc, reinterpret_cast<uint64_t *>(&smem->tma_full[stage]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem->b[stage] + i * BN * TmaAtomB, kidx * BK + i * TmaAtomB,
              bidy * BN);
        }
      }
    }
  }
  // Umma
  else if (warp_idx == 1) {
    auto idesc = Mma::idesc();
    for (uint32_t kidx = 0; kidx < total_k_blocks; advance_pipeline(kidx)) {
      smem->tma_full[stage].wait(phase);
      probe(smem->clocks[warp_idx][kidx]);
      for (uint32_t k_inner = 0; k_inner < BK; k_inner += kUmmaK) {
        auto desc_a =
            tcgen05::make_smem_desc<BM, BK>(smem->a[stage], 0, k_inner);
        auto desc_b =
            tcgen05::make_smem_desc<BN, BK>(smem->b[stage], 0, k_inner);
        // ! transposed A/B for pipelined TMA Store
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
  // Pad to warpgroup size
  else if (warp_idx < 4) {
  }
  // Epilogue
  else {
    constexpr uint32_t kEpilogueWarps = 4;
    constexpr uint32_t kTmemLoadCols = 32;
    static_assert(BM % kTmemLoadCols == 0);
    uint32_t __warp_idx = warp_idx % kEpilogueWarps;
    smem->umma_done.wait(0);
    probe(smem->clocks[warp_idx][0]);
    uint32_t n_idx = __warp_idx * 32 + lane_idx;
    cute::tma_store_fence();
    CUTLASS_PRAGMA_UNROLL
    for (uint32_t m = 0; m < BM; m += kTmemLoadCols) {
      uint32_t dregs[kTmemLoadCols];
      tcgen05::utcld<32, 32, kTmemLoadCols>(smem->tmem_ptr + m, dregs);
      cutlass::arch::fence_view_async_tmem_load();
      CUTLASS_PRAGMA_UNROLL
      for (uint32_t i = 0; i < kTmemLoadCols; ++i) {
        smem->cd[(m + i) * BN + n_idx] =
            cutlass::bfloat16_t(__uint_as_float(dregs[i]));
      }
      cutlass::arch::NamedBarrier::sync(128, 0);
      if (__warp_idx == 0 && cute::elect_one_sync()) {
        cute::SM90_TMA_STORE::copy(&cd_desc, smem->cd + m * BN, bidy * BN,
                                   bidx * BM + m);
      }
    }
    if (__warp_idx == 0) {
      if (cute::elect_one_sync()) {
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
      }
    }
    probe(smem->clocks[warp_idx][1]);
  }
  __sync_cluster();
  if (warp_idx == 0) {
    Allocator().free(smem->tmem_ptr, kNumTmemCols);
  }
#if defined(ARC_PROBE)
  if (threadIdx.x == 0) {
    auto clocks = smem->clocks;
    ARC_DLOG("|S %lu\n", clocks[7][31]);
    for (uint32_t i = 0; i < total_k_blocks; ++i) {
      ARC_DLOG(" MMA %u: %lu\n", i, smem->clocks[1][i]);
    }
    ARC_DLOG(" EPI S: %lu\n", clocks[4][0]);
    ARC_DLOG("E|: %lu\n", clocks[4][1]);
  }
#endif

#else
  CUTE_INVALID_CONTROL_PATH("Error: TCGEN05 Required");
#endif
}
} // namespace arc
