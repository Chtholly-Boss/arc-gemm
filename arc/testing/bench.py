from __future__ import annotations

import math
import os
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch


ReturnMode = str

_RETURN_MODES = {"min", "max", "mean", "median", "all"}
_DEFAULT_CACHE_SIZE_BYTES = 256 * 1024 * 1024
_KINETO_CACHE_SIZE_BYTES = int(8e9)
_PROFILER_TIME_UNITS_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1e3,
}


class _SuppressStdoutStderr:
    def __enter__(self) -> None:
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file

    def __exit__(self, *_: object) -> None:
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)
        self.outnull_file.close()
        self.errnull_file.close()


class _NoopContext:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *_: object) -> None:
        pass


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


def _parse_profiler_time_ms(time: str) -> float:
    normalized = time.replace(",", "")
    for unit, multiplier in _PROFILER_TIME_UNITS_TO_MS.items():
        if normalized.endswith(unit):
            return float(normalized[: -len(unit)]) * multiplier
    raise ValueError(f"Unsupported profiler time format: {time!r}")


def do_bench_kineto(
    fn: Callable[[], Any],
    kernel_names: str | Sequence[str],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str | None = None,
    flush_l2: bool = True,
    flush_l2_bytes: int = _KINETO_CACHE_SIZE_BYTES,
    with_multiple_kernels: bool = False,
    barrier: Callable[[], Any] | None = None,
) -> float | tuple[float, ...]:
    """
    Benchmark CUDA kernel time with ``torch.profiler``/Kineto.

    ``kernel_names`` is matched as a substring against profiler table rows. The
    returned timings are milliseconds per matching kernel call. If a tuple of
    names is provided, a tuple of timings is returned in the same order.
    """
    if num_tests <= 0:
        raise ValueError(f"num_tests must be positive, got {num_tests}")
    if flush_l2 and flush_l2_bytes <= 0:
        raise ValueError(f"flush_l2_bytes must be positive, got {flush_l2_bytes}")

    is_sequence = not isinstance(kernel_names, str)
    names = tuple(kernel_names) if is_sequence else (kernel_names,)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("kernel_names must be a string or a sequence of strings")

    # Skip profiling when running under external NVIDIA tools.
    if int(os.environ.get("DG_USE_NVIDIA_TOOLS", 0)):
        return (1.0,) * len(names) if is_sequence else 1.0

    fn()

    suppress = _SuppressStdoutStderr if suppress_kineto_output else _NoopContext
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
            acc_events=True,
        )
        with profiler:
            for _ in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(flush_l2_bytes // 4, dtype=torch.int, device="cuda").zero_()
                    if barrier is not None:
                        torch.cuda._sleep(int(2e7))
                        barrier()
                    fn()
                torch.cuda.synchronize()
                profiler.step()

    prof_lines = profiler.key_averages().table(sort_by="cuda_time_total", max_name_column_width=100).split("\n")
    if not with_multiple_kernels:
        for name in names:
            matches = sum(name in line for line in prof_lines)
            if matches > 1:
                raise AssertionError(f"Found multiple profiler rows for {name!r}: {prof_lines}")

    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    kernel_times = []
    for name in names:
        total_time_ms = 0.0
        total_calls = 0
        for line in prof_lines:
            if name not in line:
                continue
            fields = line.split()
            total_time_ms += _parse_profiler_time_ms(fields[-2]) * int(fields[-1].replace(",", ""))
            total_calls += int(fields[-1].replace(",", ""))
        kernel_times.append(total_time_ms / total_calls if total_calls > 0 else 0.0)

    return tuple(kernel_times) if is_sequence else kernel_times[0]


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
