from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch


ReturnMode = str

_RETURN_MODES = {"min", "max", "mean", "median", "all"}
_DEFAULT_CACHE_SIZE_BYTES = 256 * 1024 * 1024


def _quantile(values: Sequence[float], quantiles: Sequence[float]) -> list[float]:
    sorted_values = sorted(values)
    n = len(sorted_values)

    def compute(q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        fraction = point - lower
        return (1 - fraction) * sorted_values[lower] + fraction * sorted_values[upper]

    return [compute(q) for q in quantiles]


def _summarize_statistics(
    times: Sequence[float],
    quantiles: Sequence[float] | None,
    return_mode: ReturnMode,
) -> float | list[float]:
    if return_mode not in _RETURN_MODES:
        raise ValueError(f"return_mode must be one of {sorted(_RETURN_MODES)}, got {return_mode!r}")
    if not times:
        raise ValueError("Benchmark produced no timing samples")

    if quantiles is not None:
        ret = _quantile(times, quantiles)
        return ret[0] if len(ret) == 1 else ret
    if return_mode == "all":
        return list(times)
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
    raise AssertionError(f"unreachable return_mode: {return_mode}")


def _iter_tensors(tensors: Iterable[torch.Tensor] | torch.Tensor | None) -> Iterable[torch.Tensor]:
    if tensors is None:
        return ()
    if isinstance(tensors, torch.Tensor):
        return (tensors,)
    return tensors


def _detach_and_clear_grads(tensors: Iterable[torch.Tensor] | torch.Tensor | None) -> None:
    for tensor in _iter_tensors(tensors):
        tensor.detach_()
        tensor.requires_grad_(True)
        tensor.grad = None


def _clear_grads(tensors: Iterable[torch.Tensor] | torch.Tensor | None) -> None:
    for tensor in _iter_tensors(tensors):
        tensor.grad = None


def _repeat_for_duration(duration_ms: int | float, estimate_ms: float, fallback: int = 1000) -> int:
    if estimate_ms <= 0:
        return fallback
    return max(1, int(duration_ms / estimate_ms))


def _get_empty_cache_for_benchmark() -> torch.Tensor:
    return torch.empty(_DEFAULT_CACHE_SIZE_BYTES // 4, dtype=torch.int, device="cuda")


def _clear_cache(cache: torch.Tensor) -> None:
    cache.zero_()


def do_bench(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: Iterable[torch.Tensor] | torch.Tensor | None = None,
    quantiles: Sequence[float] | None = None,
    return_mode: ReturnMode = "mean",
) -> float | list[float]:
    """
    Benchmark ``fn`` with CUDA events.

    ``warmup`` and ``rep`` are target durations in milliseconds. The returned
    timings are also in milliseconds, matching ``triton.testing.do_bench``.
    """
    if return_mode not in _RETURN_MODES:
        raise ValueError(f"return_mode must be one of {sorted(_RETURN_MODES)}, got {return_mode!r}")

    fn()
    torch.cuda.synchronize()

    cache = _get_empty_cache_for_benchmark()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        _clear_cache(cache)
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    n_warmup = _repeat_for_duration(warmup, estimate_ms)
    n_repeat = _repeat_for_duration(rep, estimate_ms)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    for _ in range(n_warmup):
        fn()

    for i in range(n_repeat):
        _clear_grads(grad_to_none)
        _clear_cache(cache)
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_cudagraph(
    fn: Callable[[], Any],
    rep: int = 20,
    timed_runs: int | None = None,
    grad_to_none: Iterable[torch.Tensor] | torch.Tensor | None = None,
    quantiles: Sequence[float] | None = None,
    return_mode: ReturnMode = "mean",
) -> float | list[float]:
    """
    Benchmark ``fn`` by replaying a CUDA graph.

    ``rep`` is the target graph replay duration in milliseconds when
    ``timed_runs`` is not set. If ``timed_runs`` is set, it is the exact number
    of ``fn`` calls captured in the CUDA graph. The returned timings are
    milliseconds per ``fn`` call. One untimed graph replay is run before
    sampling to remove first-replay overhead.
    """
    if return_mode not in _RETURN_MODES:
        raise ValueError(f"return_mode must be one of {sorted(_RETURN_MODES)}, got {return_mode!r}")
    if timed_runs is not None and timed_runs <= 0:
        raise ValueError(f"timed_runs must be positive, got {timed_runs}")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        _detach_and_clear_grads(grad_to_none)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        n_repeat = timed_runs if timed_runs is not None else _repeat_for_duration(rep, estimate_ms)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(n_repeat):
                _clear_grads(grad_to_none)
                fn()
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()

        times = []
        for _ in range(10):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event) / n_repeat)

    return _summarize_statistics(times, quantiles, return_mode)
