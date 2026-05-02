#include "common.cuh"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/helper_macros.hpp"
#include "tcgen05.cuh"
#include "utils.cuh"
#include <cstdint>

namespace arc::experiments {
static constexpr int Sm100TmemCapacityColumns = 512;
template <uint32_t Dp, uint32_t Bits, uint32_t kIters>
CUTLASS_GLOBAL void tcgen05_cp_impl(void *sink, uint64_t *latency) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using Allocator = cute::TMEM::Allocator1Sm;
  using CpOp = tcgen05::detail::UTCCPOpSelector<Dp, Bits>;

  static_assert(kIters > 0);
  static_assert(kIters <= Sm100TmemCapacityColumns / CpOp::kCols);

  auto tick = timestamp<false>;

  extern __shared__ __align__(1024) uint8_t smem_matrix[];
  __shared__ uint32_t tmem_ptr;
  __shared__ uint64_t barrier;
  Allocator().allocate(Sm100TmemCapacityColumns, &tmem_ptr);

  if (cute::elect_one_sync()) {
    cute::initialize_barrier(barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();

  uint64_t sdesc = make_umma_smem_desc(cute::UMMA::LayoutType::SWIZZLE_128B,
                                       smem_matrix, 8 * 128, 16);
  uint64_t start = 0;

  if (cute::elect_one_sync()) {
    start = tick();
    CUTLASS_PRAGMA_UNROLL
    for (uint32_t i = 0; i < kIters; ++i) {
      tcgen05::utccp<Dp, Bits>(tmem_ptr + i * CpOp::kCols, sdesc);
    }
  }
  __syncwarp();
  cutlass::arch::umma_arrive(&barrier);
  if (cute::elect_one_sync()) {
    cute::wait_barrier(barrier, 0);
    latency[0] = start;
    latency[1] = tick();
  }
  __syncwarp();

  Allocator().free(tmem_ptr, Sm100TmemCapacityColumns);
#else
  CUTE_INVALID_CONTROL_PATH("Error: TCGEN05 Required");
#endif
}

template <uint32_t kRepeatTimes, uint32_t kLdStIters>
CUTLASS_GLOBAL void tcgen05_ld_st_impl(void *sink, uint64_t *latency) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using Allocator = cute::TMEM::Allocator1Sm;
  using LdOp = tcgen05::detail::UTCLdOpSelector<32, 32, kRepeatTimes>;
  using StOp = tcgen05::detail::UTCStOpSelector<32, 32, kRepeatTimes>;

  static_assert(LdOp::kCols == StOp::kCols);
  constexpr uint32_t kLdStCols = LdOp::kCols;
  static_assert(Sm100TmemCapacityColumns % kLdStCols == 0);
  static_assert(kLdStIters > 0);
  static_assert(kLdStIters <= Sm100TmemCapacityColumns / kLdStCols);

  __shared__ uint32_t tmem_ptr;

  uint32_t lane = cutlass::canonical_lane_idx();
  uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
  auto ldOp = tcgen05::utcld<32, 32, kRepeatTimes>;
  auto stOp = tcgen05::utcst<32, 32, kRepeatTimes>;
  auto tick = timestamp<false>;

  uint32_t r[kRepeatTimes];

  CUTLASS_PRAGMA_UNROLL
  for (uint32_t i = 0; i < kRepeatTimes; ++i) {
    r[i] = i;
  }
  if (warp_idx == 0) {
    Allocator().allocate(Sm100TmemCapacityColumns, &tmem_ptr);
  }
  __syncthreads();

  uint64_t start = tick();
  CUTLASS_PRAGMA_UNROLL
  for (uint32_t i = 0; i < kLdStIters; ++i) {
    stOp(tmem_ptr + i * kLdStCols, r);
  }
  cutlass::arch::fence_view_async_tmem_store();
  __syncthreads();
  uint64_t end = tick();
  if (cutlass::thread0()) {
    latency[0] = start;
    latency[1] = end;
  }

  uint32_t checksum = 0;
  __syncthreads();
  start = tick();
  CUTLASS_PRAGMA_UNROLL
  for (uint32_t i = 0; i < kLdStIters; ++i) {
    ldOp(tmem_ptr + i * kLdStCols, r);
  }
  cutlass::arch::fence_view_async_tmem_load();
  checksum ^= r[0];
  __syncthreads();
  end = tick();
  if (cutlass::thread0()) {
    latency[2] = start;
    latency[3] = end;
  }

  reinterpret_cast<uint32_t *>(sink)[threadIdx.x] = checksum;
  __syncthreads();

  if (warp_idx == 0) {
    Allocator().free(tmem_ptr, Sm100TmemCapacityColumns);
  }
#else
  CUTE_INVALID_CONTROL_PATH("Error: TCGEN05 Required");
#endif
}

} // namespace arc::experiments
