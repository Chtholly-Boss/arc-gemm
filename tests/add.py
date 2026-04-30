from __future__ import annotations

import typer
import torch

from arc import vecadd
from arc.testing.bench import bench_kineto


app = typer.Typer()


@app.command(no_args_is_help=True)
def check(n: int = 1024) -> None:
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(vecadd(a, b), a + b)
    typer.echo("check passed")


@app.command(no_args_is_help=True)
def bench(n: int = 1024, warm_up_runs: int = 5, timed_runs: int = 10) -> None:
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    def measure(fn, kernel_name: str) -> tuple[float, float]:
        for _ in range(warm_up_runs):
            fn()
        torch.cuda.synchronize()

        elapsed_s = bench_kineto(
            fn,
            kernel_name,
            num_tests=timed_runs,
            flush_l2=True,
        )
        bandwidth_gb_s = 3 * n * a.element_size() / elapsed_s / 1e9
        return elapsed_s, bandwidth_gb_s

    torch_time_s, torch_bandwidth_gb_s = measure(
        lambda: torch.add(a, b, out=out),
        "CUDAFunctor_add",
    )
    arc_time_s, arc_bandwidth_gb_s = measure(
        lambda: vecadd(a, b, out=out),
        "vecadd_kernel",
    )

    typer.echo(f"torch.add: {torch_time_s * 1e6:.3f} us, {torch_bandwidth_gb_s:.3f} GB/s")
    typer.echo(f"arc.vecadd: {arc_time_s * 1e6:.3f} us, {arc_bandwidth_gb_s:.3f} GB/s")


if __name__ == "__main__":
    app()
