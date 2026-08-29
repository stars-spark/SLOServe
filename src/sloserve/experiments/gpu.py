"""Concurrent, injectable GPU sampling for benchmark runs."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from types import TracebackType
from typing import Protocol, Self

from sloserve.experiments.benchmark import GpuSample


class GpuSampler(Protocol):
    """Read-only samples exposed by an asynchronous sampler context."""

    @property
    def samples(self) -> tuple[GpuSample, ...]:
        """Return the observations collected so far."""
        ...

    async def __aenter__(self) -> Self:
        """Start sampling in the background."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop sampling and await background-task cleanup."""
        ...


async def read_nvidia_smi_gpu() -> str:
    """Read one single-GPU utilization, memory, and power CSV row."""
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip() or "unknown nvidia-smi error"
        raise RuntimeError(f"nvidia-smi GPU query failed: {detail}")
    lines = [line.strip() for line in stdout.decode(errors="replace").splitlines() if line.strip()]
    if not lines:
        raise ValueError("nvidia-smi GPU query returned no rows")
    return lines[0]


class NvidiaSmiSampler:
    """Sample the first GPU concurrently, skipping unreadable ticks."""

    def __init__(
        self,
        *,
        interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        reader: Callable[[], Awaitable[str]] = read_nvidia_smi_gpu,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval_s = interval_s
        self._clock = clock
        self._sleep = sleep
        self._reader = reader
        self._samples: list[GpuSample] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def samples(self) -> tuple[GpuSample, ...]:
        """Return an immutable snapshot of collected samples."""
        return tuple(self._samples)

    async def __aenter__(self) -> Self:
        """Start a named sampling task."""
        if self._task is not None:
            raise RuntimeError("GPU sampler is already running")
        self._task = asyncio.create_task(self._sample_loop(), name="sloserve-gpu-sampler")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Cancel and await the sampling task without suppressing body failures."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _sample_loop(self) -> None:
        while True:
            try:
                raw = await self._reader()
                utilization, memory_used, power = _parse_gpu_row(raw)
                self._samples.append(
                    GpuSample(
                        timestamp_s=self._clock(),
                        utilization_percent=utilization,
                        memory_used_mib=memory_used,
                        power_w=power,
                    )
                )
            except Exception:
                # One unavailable or malformed observation is absent evidence, not a reason
                # to fail the request benchmark or invent placeholder GPU values.
                pass
            await self._sleep(self._interval_s)


def _parse_gpu_row(raw: str) -> tuple[float, float, float | None]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("GPU sample is empty")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 3:
        raise ValueError("GPU sample must contain utilization, memory, and power")

    utilization = float(fields[0])
    memory_used = float(fields[1])
    if not math.isfinite(utilization) or not math.isfinite(memory_used):
        raise ValueError("GPU utilization and memory must be finite")

    try:
        power = float(fields[2])
        if not math.isfinite(power):
            power = None
    except ValueError:
        power = None
    return utilization, memory_used, power
