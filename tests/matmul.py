from __future__ import annotations

import typer
import torch

from arc import matmul
from arc.testing.bench import do_bench


app = typer.Typer()


def _check_shape(m: int, n: int, k: int) -> None:
    assert m % 128 == 0 and n % 128 == 0 and k % 128 == 0, "matmul test expects m, n, and k to be multiples of 128"


def _make_inputs(
    m: int,
    n: int,
    k: int,
    layout: str = "nt",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout[0] not in "nt" or layout[1] not in "nt":
        raise ValueError(f"Invalid layout: {layout}")
    torch.manual_seed(seed)
    shape_a = (m, k) if layout[0] == "n" else (k, m)
    shape_b = (k, n) if layout[1] == "n" else (n, k)
    a = torch.randn(shape_a, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(shape_b, device="cuda", dtype=torch.bfloat16)
    return a, b


def _flush_l2() -> None:
    flush = torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda")
    flush.zero_()
    torch.cuda.synchronize()
    del flush


@app.command()
def check(m: int = 256, n: int = 128, k: int = 1536, seed: int = 0) -> None:
    _check_shape(m, n, k)
    a, b = _make_inputs(m, n, k, seed=seed)
    _flush_l2()
    out = matmul(a, b)
    ref = torch.mm(a, b.T)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@app.command()
def bench(
    m: int = 128,
    n: int = 128,
    k: int = 1536,
    profile: bool = False,
) -> None:
    _check_shape(m, n, k)
    a, b = _make_inputs(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    out_torch = torch.empty_like(out)
    out_arc = torch.empty_like(out)

    if profile:
        _flush_l2()
        matmul(a, b, out=out)
        torch.cuda.synchronize()
        return

    flops = 2 * m * n * k
    bytes_ = (m * k + n * k + m * n) * a.element_size()

    def measure(fn, name: str) -> float:
        elapsed_ms = do_bench(fn, warmup=100, rep=500)
        assert isinstance(elapsed_ms, float)
        tflops = flops / (elapsed_ms / 1e3) / 1e12
        gbps = bytes_ / (elapsed_ms / 1e3) / 1e9
        typer.echo(f"{name:<{24}} [{m}x{n}x{k}]: {elapsed_ms:7.3f} ms, {tflops:7.3f} TFLOP/s, {gbps:7.3f} GB/s")

    measure(lambda: torch.mm(a, b.T, out=out_torch), "torch.mm")
    measure(lambda: matmul(a, b, out=out_arc), "arc.matmul")

    # import deep_gemm

    # out_deep_gemm = torch.empty_like(out)
    # measure(lambda: deep_gemm.bf16_gemm_nt(a, b, out_deep_gemm), "deep_gemm.bf16_gemm_nt")


if __name__ == "__main__":
    app()
