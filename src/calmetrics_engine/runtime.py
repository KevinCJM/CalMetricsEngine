"""Thin asyncio adaptation for the C++ execution engine.

All planning, chunk partitioning, CPU admission, native thread/process pools,
shared-memory ownership and numerical execution live in C++. Python's executor
below only bridges a blocking native request to the application's event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from ._native import graph as _native_graph
from ._native.graph import GraphExecutionResult, PreparedGraphExecution


class AdaptiveScheduler(_native_graph.AdaptiveScheduler):
    async def execute_async(self, *args: Any, **kwargs: Any) -> GraphExecutionResult:
        """Await one native request; CPU work never executes on the event loop."""
        kwargs["async_io"] = True
        return await asyncio.to_thread(self.execute, *args, **kwargs)

    async def execute_many_async(
        self, jobs: Sequence[Mapping[str, Any]]
    ) -> list[GraphExecutionResult]:
        # Limit interface-side pending bridge calls. Native CPU admission remains
        # engine-wide across all synchronous/asynchronous callers.
        gate = asyncio.Semaphore(self.config.max_async_jobs)

        async def run(job: Mapping[str, Any]) -> GraphExecutionResult:
            async with gate:
                return await self.execute_async(**job)

        return list(await asyncio.gather(*(run(job) for job in jobs)))


def _native_program(graph):
    """Compatibility bridge for native-program inspection tools."""
    return graph._program


def _product_chunks(product_ids, starts, ends, workers):
    """Compatibility alias; weighted partitioning is implemented only in C++."""
    return _native_graph.product_chunks(product_ids, starts, ends, workers)


__all__ = ["AdaptiveScheduler", "GraphExecutionResult", "PreparedGraphExecution"]
