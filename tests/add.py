from __future__ import annotations

import typer
import torch

from arc import vecadd
from arc.testing.bench import do_bench_cudagraph


app = typer.Typer()


@app.command(no_args_is_help=True)
def check(n: int = 1024) -> None:
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(vecadd(a, b), a + b)
    typer.echo("check passed")


@app.command(no_args_is_help=True)
def bench(n: int = 1024, rep_ms: int = 20) -> None:
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    def measure(fn) -> tuple[float, float]:
        elapsed_ms = do_bench_cudagraph(
            fn,
            rep=rep_ms,
        )
        assert isinstance(elapsed_ms, float)
        elapsed_s = elapsed_ms / 1e3
        bandwidth_gb_s = 3 * n * a.element_size() / elapsed_s / 1e9
        return elapsed_s, bandwidth_gb_s

    torch_time_s, torch_bandwidth_gb_s = measure(lambda: torch.add(a, b, out=out))
    arc_time_s, arc_bandwidth_gb_s = measure(lambda: vecadd(a, b, out=out))

    typer.echo(f"torch.add: {torch_time_s * 1e6:.3f} us, {torch_bandwidth_gb_s:.3f} GB/s")
    typer.echo(f"arc.vecadd: {arc_time_s * 1e6:.3f} us, {arc_bandwidth_gb_s:.3f} GB/s")


if __name__ == "__main__":
    app()
