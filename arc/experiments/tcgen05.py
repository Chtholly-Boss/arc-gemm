from __future__ import annotations

import torch
from tvm_ffi.cpp import load_inline

from arc.jit import (
    DEFAULT_INCLUDE,
    KERNEL_PATH,
    _get_default_target_flags,
    _jit_compile_context,
    cache_once,
    get_cutlass_include_paths,
    hash_files,
)


def _source_hash() -> str:
    return hash_files(
        "csrc/tcgen05.cu",
        "include/ptx/tcgen05.cuh",
    )


@cache_once
def _load_tcgen05_cp(iters: int):
    source_hash = _source_hash()
    tcgen05_cu = KERNEL_PATH / "csrc" / "tcgen05.cu"

    def _instantiate() -> str:
        return f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include "{tcgen05_cu}"

namespace arc::experiments {{

void tcgen05_cp(tvm::ffi::TensorView sink, tvm::ffi::TensorView latency) {{
  int smem_size_used = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &smem_size_used, cudaDevAttrMaxSharedMemoryPerBlockOptin,
      sink.device().device_id));
  smem_size_used -= 4096;
  auto kernel = tcgen05_cp_impl<128, 128, {iters}>;
  CHECK_CUDA_ERROR(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_used));
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(sink.device().device_type, sink.device().device_id));
  kernel<<<1, 32, smem_size_used, stream>>>(
      sink.data_ptr(), reinterpret_cast<uint64_t *>(latency.data_ptr()));
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}}

}} // namespace arc::experiments

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tcgen05_cp, (arc::experiments::tcgen05_cp));
        """

    with _jit_compile_context():
        return load_inline(
            f"tcgen05_cp_{source_hash}_iters{iters}",
            cuda_sources=[_instantiate()],
            extra_cuda_cflags=_get_default_target_flags() + ["-DARC_PROBE"],
            extra_ldflags=["-lcuda"],
            extra_include_paths=DEFAULT_INCLUDE + get_cutlass_include_paths(),
        )


@cache_once
def _load_tcgen05_ldst(iters: int, repeat_times: int):
    source_hash = _source_hash()
    tcgen05_cu = KERNEL_PATH / "csrc" / "tcgen05.cu"

    def _instantiate() -> str:
        return f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include "{tcgen05_cu}"

namespace arc::experiments {{

void tcgen05_ld_st(tvm::ffi::TensorView sink, tvm::ffi::TensorView latency) {{
  constexpr uint32_t kRepeatTimes = {repeat_times};
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(sink.device().device_type, sink.device().device_id));
  tcgen05_ld_st_impl<kRepeatTimes, {iters}><<<1, 128, 0, stream>>>(
      sink.data_ptr(), reinterpret_cast<uint64_t *>(latency.data_ptr()));
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}}

}} // namespace arc::experiments

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tcgen05_ld_st, (arc::experiments::tcgen05_ld_st));
        """

    with _jit_compile_context():
        return load_inline(
            f"tcgen05_ldst_{source_hash}_iters{iters}_repeat{repeat_times}",
            cuda_sources=[_instantiate()],
            extra_cuda_cflags=_get_default_target_flags() + ["-DARC_PROBE"],
            extra_ldflags=["-lcuda"],
            extra_include_paths=DEFAULT_INCLUDE + get_cutlass_include_paths(),
        )


def tcgen05_cp(
    sink: torch.Tensor | None = None,
    latency: torch.Tensor | None = None,
    iters: int = 2,
) -> torch.Tensor:
    if sink is None:
        sink = torch.empty((1,), dtype=torch.int32, device="cuda")
    if latency is None:
        latency = torch.empty((1, 2), dtype=torch.int64, device=sink.device)

    _load_tcgen05_cp(iters).tcgen05_cp(sink, latency)
    return latency


def tcgen05_ld_st(
    sink: torch.Tensor | None = None,
    latency: torch.Tensor | None = None,
    iters: int = 2,
    repeat_times: int = 32,
) -> torch.Tensor:
    if sink is None:
        sink = torch.empty((128,), dtype=torch.int32, device="cuda")
    if latency is None:
        latency = torch.empty((2, 2), dtype=torch.int64, device=sink.device)

    _load_tcgen05_ldst(iters, repeat_times).tcgen05_ld_st(sink, latency)
    return latency
