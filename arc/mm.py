from __future__ import annotations

import torch
from tvm_ffi.cpp import load_inline

from .jit import (
    DEFAULT_INCLUDE,
    _get_default_target_flags,
    _jit_compile_context,
    cache_once,
    get_cutlass_include_paths,
    get_jit_cuda_arch,
    impl_hash,
    override_jit_cuda_arch,
)


def _tcgen05_arch_suffix(major: int, minor: int, suffix: str) -> str:
    if suffix:
        return suffix
    if (major, minor) in {(10, 0), (10, 1), (10, 3), (11, 0)}:
        return "f"
    return suffix


def _instantiate(tiler: tuple[int, int, int], stages: int) -> str:
    bm, bn, bk = tiler
    swizzle_k = 64
    multicast = 1
    num_threads = 256
    return f"""
// {impl_hash("mm.cuh")}
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
    make_2d_tma_desc(&cd_desc, out.data_ptr(), m, n, {bm}, {bn});
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(out.device().device_type, out.device().device_id));
    constexpr int smem_size = sizeof(SharedStorage<{bm}, {bn}, {bk}, {stages}>);
    auto kernel = gemm_tcgen05_impl<{bm}, {bn}, {bk}, {stages}, {multicast}, {num_threads}>;
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 grid(1, 1, 1);
    kernel<<<grid, {num_threads}, smem_size, stream>>>(a_desc, b_desc, cd_desc, m, n, k);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}}
}} // namespace arc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(matmul, (arc::matmul));
    """


@cache_once
def _load_matmul():
    arch = get_jit_cuda_arch()
    suffix = _tcgen05_arch_suffix(arch.major, arch.minor, arch.suffix)
    with override_jit_cuda_arch(arch.major, arch.minor, suffix):
        with _jit_compile_context():
            return load_inline(
                "matmul",
                cuda_sources=[_instantiate((128, 128, 128), 1)],
                extra_cuda_cflags=_get_default_target_flags(),
                extra_ldflags=["-lcuda"],
                extra_include_paths=DEFAULT_INCLUDE + get_cutlass_include_paths(),
            )


def matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    m, k = a.shape
    n, _k = b.shape
    assert k == _k, f"Inner dimensions must match: {k} vs {_k}"
    assert (m, n, k) == (128, 128, 1536), "TMA smoke test is pinned to m=n=128, k=1536"
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _load_matmul().matmul(a, b, out)
    return out
