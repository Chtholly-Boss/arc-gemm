#include "common.cuh"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/arch/barrier.h"
#include "ptx/tma.cuh"
#include <cstdint>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace arc::experiments {
template <bool UseGlobalTimer>
CUTLASS_GLOBAL void
tma_latency_impl(__grid_constant__ const cute::TmaDescriptor desc, uint32_t n,
                 uint32_t bytes, uint64_t *latencies) {
  auto probe = timestamp<UseGlobalTimer>;
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint64_t barrier;
  uint32_t block_row = blockIdx.x * n;
  uint64_t *block_latencies = latencies + blockIdx.x * 3 * 2;

  if (cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&desc);
    cute::initialize_barrier(barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  uint64_t start = 0, end = 0;
  if (cute::elect_one_sync()) {
    probe(start);
    cute::set_barrier_transaction_bytes(barrier, bytes);
    cute::SM90_TMA_LOAD::copy(
        &desc, &barrier,
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), smem, 0,
        block_row);
    cute::wait_barrier(barrier, 0);
    probe(end);
    block_latencies[0] = start;
    block_latencies[1] = end;

    probe(start);
    cute::set_barrier_transaction_bytes(barrier, bytes);
    cute::SM90_TMA_LOAD::copy(
        &desc, &barrier,
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), smem, 0,
        block_row);
    cute::wait_barrier(barrier, 1);
    probe(end);
    block_latencies[2] = start;
    block_latencies[3] = end;
  }
  __syncthreads();

  if (cute::elect_one_sync()) {
    cute::tma_store_fence();
    probe(start);
    cute::SM90_TMA_STORE::copy(&desc, smem, 0, block_row);
    cute::tma_store_arrive();
    cute::tma_store_wait<0>();
    probe(end);
    block_latencies[4] = start;
    block_latencies[5] = end;
  }
}

template <bool UseGlobalTimer>
void tma_latency(tvm::ffi::Tensor input, tvm::ffi::Tensor latencies) {
  uint32_t num_blocks = input.size(0);
  uint32_t num_rows = input.size(1);
  uint32_t num_cols = input.size(2);

  cute::TmaDescriptor desc;
  arc::tma::make_tma_desc(&desc, input.data_ptr(),
                          {num_blocks * num_rows, num_cols},
                          {num_rows, num_cols});
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));
  int smem_size_used = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &smem_size_used, cudaDevAttrMaxSharedMemoryPerBlockOptin,
      input.device().device_id));
  smem_size_used -= 4096; // reserve some shared memory for other usages

  CHECK_CUDA_ERROR(cudaFuncSetAttribute(
      tma_latency_impl<UseGlobalTimer>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_used));
  tma_latency_impl<UseGlobalTimer><<<num_blocks, 32, smem_size_used, stream>>>(
      desc, num_rows, num_rows * num_cols,
      static_cast<uint64_t *>(latencies.data_ptr()));
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}
} // namespace arc::experiments
