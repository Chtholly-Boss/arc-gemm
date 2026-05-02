from __future__ import annotations

from enum import Enum
from statistics import mean
from typing import Annotated

import torch
import typer

from arc.experiments import tma_latency


L2_FLUSH_BYTES = int(8e9)
app = typer.Typer(no_args_is_help=True)


class TimerMode(str, Enum):
    ns = "ns"
    clock64 = "clock64"


def _flush_l2(buffer: torch.Tensor) -> None:
    buffer.zero_()
    torch.cuda.synchronize()


def _stats(values: list[int], unit: str) -> str:
    return f"avg={mean(values):.1f} {unit} [min,max] = [{min(values)}, {max(values)}]"


@app.command(no_args_is_help=True)
def main(
    n: Annotated[int, typer.Option("--n")],
    num_blocks: Annotated[int, typer.Option("--num-blocks")] = 2,
    width_bytes: Annotated[int, typer.Option("--width-bytes")] = 128,
    mode: Annotated[TimerMode, typer.Option("--mode")] = TimerMode.ns,
    warm_up_runs: Annotated[int, typer.Option("--warm-up-runs")] = 1,
    timed_runs: Annotated[int, typer.Option("--timed-runs")] = 10,
) -> None:
    input_tensor = torch.empty((num_blocks, n, width_bytes), device="cuda", dtype=torch.uint8)
    latencies = torch.empty((num_blocks, 3, 2), device="cuda", dtype=torch.int64)
    flush_buffer = torch.empty(L2_FLUSH_BYTES // 4, device="cuda", dtype=torch.int32)

    for _ in range(warm_up_runs):
        tma_latency(input_tensor, latencies, mode=mode.value)

    samples: list[list[list[int]]] = []
    for _ in range(timed_runs):
        _flush_l2(flush_buffer)
        samples.append(tma_latency(input_tensor, latencies, mode=mode.value).cpu().tolist())

    unit = "ns" if mode == TimerMode.ns else "cycles"
    typer.echo(f"TMA mode={mode.value}, B={num_blocks}, N={n}, W={width_bytes}B, bytes/blk={n * width_bytes}, timed_runs={timed_runs}")
    for b in range(num_blocks):
        load_cycles = [int(sample[b][0][1] - sample[b][0][0]) for sample in samples]
        load_l2_hit_cycles = [int(sample[b][1][1] - sample[b][1][0]) for sample in samples]
        store_cycles = [int(sample[b][2][1] - sample[b][2][0]) for sample in samples]
        typer.echo(f"#B{b} Load (Global):       {_stats(load_cycles, unit)}")
        typer.echo(f"#B{b} Load (L2 hit): {_stats(load_l2_hit_cycles, unit)}")
        typer.echo(f"#B{b} Store:      {_stats(store_cycles, unit)}")


if __name__ == "__main__":
    app()
