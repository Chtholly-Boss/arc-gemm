from __future__ import annotations

import deep_gemm
import torch
import typer

from arc.math import count_bytes, symm_square_rel_L2_error
from arc.testing.bench import do_bench
from arc.testing.quant import TensorPair, per_token_cast_to_fp4, per_token_cast_to_fp8


app = typer.Typer(no_args_is_help=True)


def generate_data(
    m: int,
    n: int,
    k: int,
    *,
    seed: int,
) -> tuple[TensorPair, TensorPair, torch.Tensor]:
    if k % 2 != 0:
        raise ValueError("fp8 x packed-fp4 GEMM requires an even K dimension")

    torch.manual_seed(seed)
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=128)
    b = per_token_cast_to_fp4(b_ref, use_ue8m0=True, gran_k=32)

    ref = a_ref.float() @ b_ref.float().T
    return a, b, ref.to(torch.bfloat16)


@app.command()
def check(
    m: int = 128,
    n: int = 128,
    k: int = 128,
    seed: int = 0,
    tol: float = 1e-2,
) -> None:
    a, b, ref = generate_data(m, n, k, seed=seed)

    out = torch.empty_like(ref)
    try:
        deep_gemm.fp8_fp4_gemm_nt(a, b, out, recipe_a=(1, 128), recipe_b=(1, 32))
    except Exception as exc:
        raise RuntimeError(f"fp8/fp4 GEMM failed, m={m}, n={n}, k={k}") from exc
    torch.cuda.synchronize()

    diff = symm_square_rel_L2_error(out, ref)
    assert diff < tol, f"fp8/fp4 GEMM diff {diff:.6g} exceeded {tol:.6g}, m={m}, n={n}, k={k}"
    typer.echo(f"fp8/fp4 GEMM passed, m={m}, n={n}, k={k}, diff={diff:.6g}")


@app.command()
def bench(
    m: int = 128,
    n: int = 128,
    k: int = 128,
    seed: int = 0,
    profile: bool = False,
) -> None:
    a, b, ref = generate_data(m, n, k, seed=seed)
    out = torch.empty_like(ref)

    def run() -> None:
        deep_gemm.fp8_fp4_gemm_nt(a, b, out, recipe_a=(1, 128), recipe_b=(1, 32))

    if profile:
        run()
        torch.cuda.synchronize()
        return

    flops = 2 * m * n * k
    bytes_ = count_bytes(a, b, out)
    elapsed_ms = do_bench(run, warmup=100, rep=500)
    assert isinstance(elapsed_ms, float)
    tflops = flops / (elapsed_ms / 1e3) / 1e12
    gbps = bytes_ / (elapsed_ms / 1e3) / 1e9
    typer.echo(
        f"{'deep_gemm.fp8_fp4_gemm_nt':<{24}} [{m}x{n}x{k}]: "
        f"{elapsed_ms:7.3f} ms, {tflops:7.3f} TFLOP/s, {gbps:7.3f} GB/s"
    )


if __name__ == "__main__":
    app()
