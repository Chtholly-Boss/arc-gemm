from __future__ import annotations

import torch

from arc.jit import cache_once, hash_files, load_jit


@cache_once
def _load_tma_latency():
    source_hash = hash_files(
        "csrc/tma.cu",
        "include/utils.cuh",
        "include/common.cuh",
    )
    return load_jit(
        "tma_latency",
        source_hash,
        cuda_files=["tma.cu"],
        cuda_wrappers=[
            ("tma_latency_clock64", "arc::experiments::tma_latency<false>"),
            ("tma_latency_ns", "arc::experiments::tma_latency<true>"),
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=["-DARC_PROBE"],
        extra_ldflags=["-lcuda"],
    )


def tma_latency(input: torch.Tensor, latencies: torch.Tensor | None = None, mode="ns") -> torch.Tensor:
    if mode not in ("clock64", "ns"):
        raise ValueError(f"Invalid mode: {mode}, expected 'clock64' or 'ns'")
    mod = _load_tma_latency()
    kernel = mod.tma_latency_ns if mode == "ns" else mod.tma_latency_clock64

    if latencies is None:
        latencies = torch.empty((input.shape[0], 3, 2), dtype=torch.int64, device=input.device)
    kernel(input, latencies)

    return latencies
