from __future__ import annotations

from typing import TypeAlias

import torch

from arc.math import align


TensorPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float or x.size(-1) % 4 != 0:
        raise ValueError("UE8M0 packing expects float scales with last dimension divisible by 4")
    if not bool((x.view(torch.int) & ((1 << 23) - 1) == 0).all()):
        raise ValueError("UE8M0 packing expects power-of-two float scale factors")
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> TensorPair:
    if x.dim() != 2:
        raise ValueError("per-token fp8 casting expects a 2D tensor")
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / FP8_E4M3_MAX
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    return x_fp8, pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(FP4_E2M1_MAX)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    return (code | (sign.to(torch.uint8) << 3)).view(torch.int8)


def per_token_cast_to_fp4(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 32,
    use_packed_ue8m0: bool = False,
) -> TensorPair:
    if x.dim() != 2:
        raise ValueError("per-token fp4 casting expects a 2D tensor")
    m, n = x.shape
    if n % 2 != 0:
        raise ValueError("packed fp4 casting expects an even K dimension")
    if use_packed_ue8m0 and not use_ue8m0:
        raise ValueError("packed UE8M0 scales require use_ue8m0=True")
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / FP4_E2M1_MAX
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    codes = _quantize_to_fp4_e2m1(x_view * (1.0 / sf.unsqueeze(2))).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


__all__ = [
    "FP4_E2M1_MAX",
    "FP8_E4M3_MAX",
    "TensorPair",
    "ceil_to_ue8m0",
    "pack_ue8m0_to_int",
    "per_token_cast_to_fp4",
    "per_token_cast_to_fp8",
]
