from __future__ import annotations

from statistics import mean
from typing import Annotated

import torch
import typer

from arc.experiments import tcgen05_cp, tcgen05_ld_st


app = typer.Typer(no_args_is_help=False)

def _cp_metrics(iters: int) -> list[tuple[str, int]]:
    return [("cp.128x128b", iters * 128 * 16)]


def _ld_st_metrics(iters: int, repeat_times: int) -> list[tuple[str, int]]:
    bytes_per_op = 4 * 32 * repeat_times * 4
    return [
        (f"st.32x32b.x{repeat_times}", iters * bytes_per_op),
        (f"ld.32x32b.x{repeat_times}", iters * bytes_per_op),
    ]


def _flush_l2(buffer: torch.Tensor) -> None:
    buffer.zero_()
    torch.cuda.synchronize()


def _samples_to_cycles(samples: list[list[list[int]]], metric_idx: int) -> list[int]:
    return [int(sample[metric_idx][1] - sample[metric_idx][0]) for sample in samples]


def _report(title: str, metrics: list[tuple[str, int]], samples: list[list[list[int]]]) -> None:
    typer.echo(title)
    for idx, (name, bytes_moved) in enumerate(metrics):
        cycles = _samples_to_cycles(samples, idx)
        avg_cycles = mean(cycles)
        b_per_cycle = bytes_moved / avg_cycles
        typer.echo(
            f"{name:24s}: avg={avg_cycles:8.1f} cycles "
            f"[min,max]=[{min(cycles)}, {max(cycles)}], "
            f"{b_per_cycle:7.2f} B/cycle"
        )


@app.command()
def main(
    warm_up_runs: Annotated[int, typer.Option("--warm-up-runs")] = 3,
    timed_runs: Annotated[int, typer.Option("--timed-runs")] = 10,
    iters: Annotated[int, typer.Option("--iters")] = 2,
    repeat_times: Annotated[int, typer.Option("--repeat-times")] = 32,
    flush_l2: Annotated[bool, typer.Option("--flush-l2/--no-flush-l2")] = False,
) -> None:
    sink = torch.empty((128,), device="cuda", dtype=torch.int32)
    cp_latency = torch.empty((1, 2), device="cuda", dtype=torch.int64)
    ld_st_latency = torch.empty((2, 2), device="cuda", dtype=torch.int64)
    flush_buffer = torch.empty((64 << 20) // 4, device="cuda", dtype=torch.int32)

    for _ in range(warm_up_runs):
        tcgen05_cp(sink, cp_latency, iters=iters)
        tcgen05_ld_st(sink, ld_st_latency, iters=iters, repeat_times=repeat_times)

    cp_samples: list[list[list[int]]] = []
    ld_st_samples: list[list[list[int]]] = []
    for _ in range(timed_runs):
        if flush_l2:
            _flush_l2(flush_buffer)
        cp_samples.append(tcgen05_cp(sink, cp_latency, iters=iters).cpu().tolist())

        if flush_l2:
            _flush_l2(flush_buffer)
        ld_st_samples.append(
            tcgen05_ld_st(sink, ld_st_latency, iters=iters, repeat_times=repeat_times)
            .cpu()
            .tolist()
        )

    typer.echo(
        f"tcgen05 microbench, timed_runs={timed_runs}, "
        f"iters={iters}, repeat_times={repeat_times}"
    )
    _report("Shared memory -> Tensor Memory", _cp_metrics(iters), cp_samples)
    _report(
        "Tensor Memory <-> registers",
        _ld_st_metrics(iters, repeat_times),
        ld_st_samples,
    )


if __name__ == "__main__":
    app()
