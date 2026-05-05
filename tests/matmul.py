from __future__ import annotations

import typer
import torch

from arc import matmul
from arc.math import count_bytes
from arc.testing.bench import do_bench_kineto


app = typer.Typer()


def generate_data(
    m: int,
    n: int,
    k: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    ref = torch.mm(a.float(), b.float().T).to(torch.bfloat16)
    return a, b, ref


def _flush_l2() -> None:
    flush = torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda")
    flush.zero_()
    torch.cuda.synchronize()
    del flush


@app.command(no_args_is_help=True)
def check(
    m: int,
    n: int,
    k: int,
    seed: int = 0,
) -> None:
    a, b, ref = generate_data(m, n, k, seed=seed)
    _flush_l2()
    out = matmul(a, b)
    torch.testing.assert_close(out.float(), ref)


@app.command(no_args_is_help=True)
def bench(
    m: int,
    n: int,
    k: int,
    once: bool = False,
) -> None:
    a, b, _ = generate_data(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    out_torch = torch.empty_like(out)
    out_arc = torch.empty_like(out)
    out_deep_gemm = torch.empty_like(out)

    if once:
        _flush_l2()
        matmul(a, b, out=out)
        torch.cuda.synchronize()
        return

    flops = 2 * m * n * k
    bytes_ = count_bytes(a, b, out)

    def measure(fn, kernel_names: str | tuple[str, ...], name: str) -> None:
        elapsed = do_bench_kineto(fn, kernel_names, suppress_kineto_output=True)
        elapsed_ms = sum(elapsed) if isinstance(elapsed, tuple) else elapsed
        tflops = flops / (elapsed_ms / 1e3) / 1e12
        gbps = bytes_ / (elapsed_ms / 1e3) / 1e9
        typer.echo(f"{name:<{24}} [{m}x{n}x{k}]: {elapsed_ms:7.3f} ms, {tflops:7.3f} TFLOP/s, {gbps:7.3f} GB/s")

    measure(lambda: torch.mm(a, b.T, out=out_torch), ("nvjet", "reduce"), "torch.mm")
    measure(lambda: matmul(a, b, out=out_arc), "arc::gemm_tcgen05_impl", "arc.matmul")

    import deep_gemm

    measure(lambda: deep_gemm.bf16_gemm_nt(a, b, out_deep_gemm), "bf16_gemm", "deep_gemm.bf16")


if __name__ == "__main__":
    app()
