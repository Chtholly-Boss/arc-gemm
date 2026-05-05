from __future__ import annotations

import torch
from tvm_ffi.cpp import load_inline

from .jit import (
    DEFAULT_INCLUDE,
    _get_default_target_flags,
    _jit_compile_context,
    cache_once,
    get_env,
    get_cutlass_include_paths,
    hash_files,
)


def _instantiate(tiler: tuple[int, int, int], stages: int) -> str:
    bm, bn, bk = tiler
    swizzle_k = 64
    multicast = 1
    num_threads = 256
    return f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <mm.cuh>

namespace arc {{

void matmul(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView out) {{
    auto m = static_cast<uint32_t>(a.size(0));
    auto k = static_cast<uint32_t>(a.size(1));
    auto n = static_cast<uint32_t>(b.size(0)); // K-Major
    cute::TmaDescriptor a_desc, b_desc, cd_desc;
    make_2d_tma_desc(&a_desc, a.data_ptr(), m, k, {bm}, {swizzle_k}, CU_TENSOR_MAP_SWIZZLE_128B);
    make_2d_tma_desc(&b_desc, b.data_ptr(), n, k, {bn}, {swizzle_k}, CU_TENSOR_MAP_SWIZZLE_128B);
    make_2d_tma_desc(&cd_desc, out.data_ptr(), m, n, 32, {bn}, CU_TENSOR_MAP_SWIZZLE_NONE);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(out.device().device_type, out.device().device_id));
    using SmemStorage = SharedStorage<{bm}, {bn}, {bk}, {stages}, cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t>;
    constexpr int smem_size = sizeof(SmemStorage);
    auto kernel = gemm_tcgen05_impl<{bm}, {bn}, {bk}, {stages}, {multicast}, {num_threads}>;
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 grid((m + {bm} - 1) / {bm}, (n + {bn} - 1) / {bn}, 1);
    kernel<<<grid, {num_threads}, smem_size, stream>>>(a_desc, b_desc, cd_desc, m, n, k);
}}
}} // namespace arc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(matmul, (arc::matmul));
    """


@cache_once
def _load_matmul():
    source_hash = hash_files(
        "include/mm.cuh",
        "include/utils.cuh",
        "include/common.cuh",
    )
    probe = get_env("ARC_PROBE")
    with _jit_compile_context():
        return load_inline(
            f"matmul_{source_hash}{'_probe' if probe else ''}",
            cuda_sources=[_instantiate((128, 128, 128), 3)],
            extra_cuda_cflags=_get_default_target_flags() + (["-DARC_PROBE"] if probe else []),
            extra_ldflags=["-lcuda"],
            extra_include_paths=DEFAULT_INCLUDE + get_cutlass_include_paths(),
        )


def matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    m, k = a.shape
    n, _k = b.shape
    assert k == _k, f"Inner dimensions must match: {k} vs {_k}"
    assert m % 128 == 0 and n % 128 == 0 and k % 128 == 0, "matmul expects m, n, and k to be multiples of 128"
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _load_matmul().matmul(a, b, out)
    return out
