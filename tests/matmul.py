from __future__ import annotations

import typer
import torch

from arc import matmul
from arc.testing.bench import bench_kineto


app = typer.Typer()


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


@app.command()
def check(m: int = 128, n: int = 128, k: int = 1536, seed: int = 0) -> None:
    assert (m, n, k) == (128, 128, 1536), "matmul smoke check is pinned to m=n=128, k=1536"
    a, b = _make_inputs(m, n, k, seed=seed)
    out = matmul(a, b)
    ref = torch.mm(a, b.T)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@app.command()
def bench(
    m: int = 128,
    n: int = 128,
    k: int = 1536,
    warm_up_runs: int = 5,
    timed_runs: int = 30,
) -> None:
    assert (m, n, k) == (128, 128, 1536), "matmul benchmark is pinned to m=n=128, k=1536"
    a, b = _make_inputs(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def measure(fn, kernel_name: str) -> tuple[float, float]:
        for _ in range(warm_up_runs):
            fn()
        torch.cuda.synchronize()

        elapsed_s = bench_kineto(
            fn,
            kernel_name,
            num_tests=timed_runs,
            suppress_kineto_output=True,
            flush_l2=True,
        )
        if elapsed_s <= 0:
            raise RuntimeError(f"Kineto did not find kernel matching {kernel_name!r}")
        tflops = 2 * m * n * k / elapsed_s / 1e12
        return elapsed_s, tflops

    arc_time_s, arc_tflops = measure(lambda: matmul(a, b, out=out), "gemm_tcgen05_impl")
    typer.echo(f"arc.matmul [{m}x{n}x{k}]: {arc_time_s * 1e6:.3f} us, {arc_tflops:.3f} TFLOP/s")


if __name__ == "__main__":
    app()
