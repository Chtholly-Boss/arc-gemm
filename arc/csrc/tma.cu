#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/arch/barrier.h"
#include "utils.cuh"
#include <cstdint>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

inline void make_2d_tma_desc_u8(cute::TmaDescriptor *desc, void *ptr,
                                uint64_t height, uint64_t width,
                                uint32_t box_height, uint32_t box_width) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {width, height};
  uint64_t globalStrides[rank - 1] = {width};
  uint32_t boxDim[rank] = {box_width, box_height};
  uint32_t elementStrides[rank] = {1, 1};
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
  // ! boxDim array, which specifies number of elements to be traversed along
  // each of the tensorRank dimensions, must be non-zero and less than or equal
  // to 256
  CHECK_CUDRV_ERROR(cuTensorMapEncodeTiled(
      desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, ptr, globalDim, globalStrides,
      boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

namespace arc::experiments {
template <bool UseGlobalTimer>
CUTLASS_GLOBAL void
tma_latency_impl(__grid_constant__ const cute::TmaDescriptor desc, uint32_t n,
                 uint32_t bytes, uint64_t *latencies) {
  auto record = timestamp<UseGlobalTimer>;
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

  if (cute::elect_one_sync()) {
    uint64_t start = record();
    cute::set_barrier_transaction_bytes(barrier, bytes);
    cute::SM90_TMA_LOAD_2D::copy(
        &desc, &barrier,
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), smem, 0,
        block_row);
    cute::wait_barrier(barrier, 0);
    uint64_t end = record();
    block_latencies[0] = start;
    block_latencies[1] = end;

    start = record();
    cute::set_barrier_transaction_bytes(barrier, bytes);
    cute::SM90_TMA_LOAD_2D::copy(
        &desc, &barrier,
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), smem, 0,
        block_row);
    cute::wait_barrier(barrier, 1);
    end = record();
    block_latencies[2] = start;
    block_latencies[3] = end;
  }
  __syncthreads();

  if (cute::elect_one_sync()) {
    cute::tma_store_fence();
    uint64_t start = record();
    cute::SM90_TMA_STORE_2D::copy(&desc, smem, 0, block_row);
    cute::tma_store_arrive();
    cute::tma_store_wait<0>();
    uint64_t end = record();
    block_latencies[4] = start;
    block_latencies[5] = end;
  }
}

template <bool UseGlobalTimer>
void tma_latency(tvm::ffi::Tensor input, tvm::ffi::Tensor latencies) {
  auto num_sm = input.size(0);
  auto num_rows = input.size(1);
  auto num_cols = input.size(2);

  cute::TmaDescriptor desc;
  make_2d_tma_desc_u8(&desc, input.data_ptr(), num_sm * num_rows, num_cols,
                      num_rows, num_cols);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));
  int smem_size_used = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &smem_size_used, cudaDevAttrMaxSharedMemoryPerBlockOptin,
      input.device().device_id));
  smem_size_used -= 4096; // reserve some shared memory for other usages

  CHECK_CUDA_ERROR(cudaFuncSetAttribute(
      tma_latency_impl<UseGlobalTimer>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size_used));
  tma_latency_impl<UseGlobalTimer><<<num_sm, 32, smem_size_used, stream>>>(
      desc, num_rows, num_rows * num_cols,
      static_cast<uint64_t *>(latencies.data_ptr()));
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}
} // namespace arc::experiments
