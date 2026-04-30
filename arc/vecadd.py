from __future__ import annotations

import torch
from tvm_ffi.cpp import load_inline

from .jit import (
    DEFAULT_INCLUDE,
    _get_default_target_flags,
    _jit_compile_context,
    cache_once,
)


def _instantiate(dtype: torch.dtype, num_threads: int) -> str:
    if dtype == torch.float32:
        dtype = "float"
    elif dtype == torch.int32:
        dtype = "int"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <add.cuh>

namespace arc {{

void vecadd(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView out) {{
    cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(out.device().device_type, out.device().device_id));
    vecadd_kernel<<<(out.numel() + {num_threads - 1}) / {num_threads}, {num_threads}, 0, stream>>>(
        static_cast<const {dtype}*>(a.data_ptr()),
        static_cast<const {dtype}*>(b.data_ptr()),
        static_cast<{dtype}*>(out.data_ptr()),
        out.numel());
}}

}} // namespace arc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vecadd, (arc::vecadd));
    """


@cache_once
def _load_vecadd():
    with _jit_compile_context():
        return load_inline(
            "vecadd",
            cuda_sources=[_instantiate(torch.float32, 256)],
            extra_cuda_cflags=_get_default_target_flags(),
            extra_include_paths=DEFAULT_INCLUDE,
        )


def vecadd(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(a)

    _load_vecadd().vecadd(a, b, out)
    return out
