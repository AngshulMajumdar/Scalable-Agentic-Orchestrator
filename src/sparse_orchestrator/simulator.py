"""Wave and asynchronous event simulators."""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .metrics import (
    resource_time_lower_bound,
    unit_wave_lower_bound,
    utilization_summary,
    verify_dispatch,
)
from .model import (
    AgentSet,
    Provider,
    ScheduleTrace,
    SimulationResult,
    ValidationError,
)
from .schedulers import BatchScheduler
from .storage import ActiveSet


@dataclass(slots=True)
class _Accumulator:
    scheduler_time_s: float
    usage_integral: np.ndarray
    dispatches: int
    completed: int


class Simulator:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.config.validate()

    def run(
        self,
        agents: AgentSet,
        provider: Provider,
        scheduler: BatchScheduler,
    ) -> SimulationResult:
        agents.validate_against(provider)
        mode = self.config.mode
        if mode == "auto":
            mode = "waves" if agents.unit_duration else "events"
        scheduler.reset(agents, provider)
        try:
            if mode == "waves":
                return self._run_waves(agents, provider, scheduler)
            return self._run_events(agents, provider, scheduler)
        finally:
            scheduler.close()

    def _run_waves(
        self,
        agents: AgentSet,
        provider: Provider,
        scheduler: BatchScheduler,
    ) -> SimulationResult:
        if not agents.unit_duration:
            raise ValidationError("wave simulation requires equal durations")
        wall_start = time.perf_counter()
        duration = float(agents.durations[0])
        active = ActiveSet(agents.n_agents)
        trace = ScheduleTrace.allocate(agents.n_agents) if self.config.record_trace else None
        usage_integral = np.zeros(provider.n_resources, dtype=np.float64)
        scheduler_time = 0.0
        dispatches = 0
        wave = 0

        while not active.empty:
            if self.config.max_dispatches is not None and dispatches >= self.config.max_dispatches:
                raise RuntimeError("maximum dispatch count exceeded")
            remaining = provider.capacity.copy()
            selected_this_wave = 0
            while True:
                batch = scheduler.select(agents, provider, active, remaining)
                scheduler_time += batch.selector_time_s
                if batch.indices.size == 0:
                    break
                if self.config.validate_every_dispatch:
                    verify_dispatch(agents.demands, batch.indices, remaining)
                if np.unique(batch.indices).size != batch.indices.size:
                    raise ValidationError("scheduler returned duplicate indices in one batch")
                if not np.all(active.mask[batch.indices]):
                    raise ValidationError("scheduler returned an inactive agent")
                if trace is not None:
                    trace.start_times[batch.indices] = wave * duration
                    trace.finish_times[batch.indices] = (wave + 1) * duration
                    trace.wave_ids[batch.indices] = wave
                active.remove(batch.indices)
                remaining -= np.sum(agents.demands[batch.indices], axis=0, dtype=np.float64)
                remaining[np.abs(remaining) < 1e-10] = 0.0
                if np.any(remaining < -1e-7):
                    raise ValidationError("negative capacity after dispatch")
                selected_this_wave += int(batch.indices.size)
                dispatches += 1
            if selected_this_wave == 0:
                # Every individual agent was validated against full capacity, so
                # this indicates a scheduler/backend error rather than infeasibility.
                first = int(active.indices()[0])
                raise RuntimeError(
                    f"scheduler {scheduler.name} made no progress on wave {wave}; "
                    f"first active agent is {first}"
                )
            used = provider.capacity - remaining
            usage_integral += used * duration
            wave += 1

        makespan = wave * duration
        lower_bound = unit_wave_lower_bound(agents, provider) * duration
        mean_util, min_util, p95_util, per_resource = utilization_summary(
            usage_integral, provider, makespan
        )
        if trace is not None:
            trace.validate_complete()
        diagnostics = scheduler.stats.as_dict()
        diagnostics.update(
            {
                "simulation_mode": "waves",
                "waves": wave,
                "utilization_by_resource": ",".join(f"{x:.8f}" for x in per_resource),
            }
        )
        return SimulationResult(
            method=scheduler.name,
            n_agents=agents.n_agents,
            n_resources=agents.n_resources,
            makespan=makespan,
            lower_bound=lower_bound,
            normalized_makespan=makespan / lower_bound,
            scheduler_time_s=scheduler_time,
            wall_time_s=time.perf_counter() - wall_start,
            dispatches=dispatches,
            mean_utilization=mean_util,
            min_utilization=min_util,
            p95_utilization=p95_util,
            completed=agents.n_agents,
            valid=True,
            diagnostics=diagnostics,
            trace=trace,
        )

    def _run_events(
        self,
        agents: AgentSet,
        provider: Provider,
        scheduler: BatchScheduler,
    ) -> SimulationResult:
        wall_start = time.perf_counter()
        active = ActiveSet(agents.n_agents)
        trace = ScheduleTrace.allocate(agents.n_agents) if self.config.record_trace else None
        remaining = provider.capacity.copy()
        running: list[tuple[float, int]] = []
        scheduler_time = 0.0
        dispatches = 0
        completed = 0
        now = 0.0
        previous_time = 0.0
        usage_integral = np.zeros(provider.n_resources, dtype=np.float64)
        event_tolerance = self.config.event_tolerance

        while completed < agents.n_agents:
            while not active.empty:
                if self.config.max_dispatches is not None and dispatches >= self.config.max_dispatches:
                    raise RuntimeError("maximum dispatch count exceeded")
                batch = scheduler.select(agents, provider, active, remaining)
                scheduler_time += batch.selector_time_s
                if batch.indices.size == 0:
                    break
                if self.config.validate_every_dispatch:
                    verify_dispatch(agents.demands, batch.indices, remaining)
                if not np.all(active.mask[batch.indices]):
                    raise ValidationError("scheduler returned inactive agents")
                used = np.sum(agents.demands[batch.indices], axis=0, dtype=np.float64)
                remaining -= used
                remaining[np.abs(remaining) < 1e-10] = 0.0
                if np.any(remaining < -1e-7):
                    raise ValidationError("negative capacity after event dispatch")
                active.remove(batch.indices)
                for index in batch.indices:
                    i = int(index)
                    finish = now + float(agents.durations[i])
                    heapq.heappush(running, (finish, i))
                    if trace is not None:
                        trace.start_times[i] = now
                        trace.finish_times[i] = finish
                        trace.wave_ids[i] = dispatches
                dispatches += 1

            if not running:
                if active.empty:
                    break
                first = int(active.indices()[0])
                raise RuntimeError(
                    f"scheduler {scheduler.name} deadlocked at t={now}; first active={first}"
                )

            next_time = running[0][0]
            elapsed = next_time - previous_time
            usage_integral += (provider.capacity - remaining) * elapsed
            previous_time = next_time
            now = next_time
            finished: list[int] = []
            while running and abs(running[0][0] - now) <= event_tolerance:
                _, index = heapq.heappop(running)
                finished.append(index)
            if finished:
                released = np.sum(agents.demands[np.asarray(finished)], axis=0, dtype=np.float64)
                remaining += released
                # Limit accumulated round-off without hiding substantive errors.
                remaining = np.minimum(remaining, provider.capacity)
                completed += len(finished)

        makespan = now
        lower_bound = resource_time_lower_bound(agents, provider)
        mean_util, min_util, p95_util, per_resource = utilization_summary(
            usage_integral, provider, makespan
        )
        if trace is not None:
            trace.validate_complete()
        diagnostics = scheduler.stats.as_dict()
        diagnostics.update(
            {
                "simulation_mode": "events",
                "utilization_by_resource": ",".join(f"{x:.8f}" for x in per_resource),
            }
        )
        return SimulationResult(
            method=scheduler.name,
            n_agents=agents.n_agents,
            n_resources=agents.n_resources,
            makespan=makespan,
            lower_bound=lower_bound,
            normalized_makespan=makespan / lower_bound,
            scheduler_time_s=scheduler_time,
            wall_time_s=time.perf_counter() - wall_start,
            dispatches=dispatches,
            mean_utilization=mean_util,
            min_utilization=min_util,
            p95_utilization=p95_util,
            completed=completed,
            valid=completed == agents.n_agents,
            diagnostics=diagnostics,
            trace=trace,
        )


__all__ = ["Simulator"]
