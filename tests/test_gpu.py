"""Offline tests for concurrent GPU sampling."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from sloserve.experiments.gpu import NvidiaSmiSampler


def _controlled_sleep(after_ticks: int) -> tuple[Callable[[float], Awaitable[None]], asyncio.Event]:
    ready = asyncio.Event()
    tick_count = 0

    async def sleep(_: float) -> None:
        nonlocal tick_count
        tick_count += 1
        if tick_count >= after_ticks:
            ready.set()
            await asyncio.get_running_loop().create_future()
        await asyncio.sleep(0)

    return sleep, ready


def test_sampler_uses_injected_clock_parses_missing_power_and_cleans_up() -> None:
    async def scenario() -> None:
        rows = iter(("45, 6136, 60.50", "12, 6100, [N/A]"))
        timestamps = iter((101.0, 102.0))

        async def reader() -> str:
            return next(rows)

        sleep, ready = _controlled_sleep(after_ticks=2)
        sampler = NvidiaSmiSampler(
            interval_s=0.5,
            clock=lambda: next(timestamps),
            sleep=sleep,
            reader=reader,
        )

        async with sampler:
            await ready.wait()
            assert [sample.timestamp_s for sample in sampler.samples] == [101.0, 102.0]
            assert [sample.utilization_percent for sample in sampler.samples] == [45.0, 12.0]
            assert [sample.memory_used_mib for sample in sampler.samples] == [6136.0, 6100.0]
            assert [sample.power_w for sample in sampler.samples] == [60.5, None]

        assert all(task.get_name() != "sloserve-gpu-sampler" for task in asyncio.all_tasks())

    asyncio.run(scenario())


def test_sampler_skips_reader_failure_without_crashing() -> None:
    async def scenario() -> None:
        attempts = 0

        async def reader() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("offline fake failure")
            return "20, 2048, invalid-power"

        sleep, ready = _controlled_sleep(after_ticks=2)
        sampler = NvidiaSmiSampler(
            interval_s=0.25,
            clock=lambda: 7.0,
            sleep=sleep,
            reader=reader,
        )

        async with sampler:
            await ready.wait()

        assert attempts == 2
        assert len(sampler.samples) == 1
        assert sampler.samples[0].timestamp_s == 7.0
        assert sampler.samples[0].power_w is None

    asyncio.run(scenario())
