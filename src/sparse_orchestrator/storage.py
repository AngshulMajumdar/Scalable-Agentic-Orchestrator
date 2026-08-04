"""Storage and active-set utilities for large explicit agent arrays."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .model import AgentSet, IntArray, ValidationError


@dataclass(frozen=True, slots=True)
class MemmapManifest:
    root: Path
    n_agents: int
    n_resources: int
    demands_dtype: str
    durations_dtype: str

    @property
    def demands_path(self) -> Path:
        return self.root / "demands.dat"

    @property
    def durations_path(self) -> Path:
        return self.root / "durations.dat"

    @property
    def ids_path(self) -> Path:
        return self.root / "ids.dat"

    @property
    def arrival_order_path(self) -> Path:
        return self.root / "arrival_order.dat"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def as_dict(self) -> dict[str, object]:
        return {
            "n_agents": self.n_agents,
            "n_resources": self.n_resources,
            "demands_dtype": self.demands_dtype,
            "durations_dtype": self.durations_dtype,
        }


def save_memmap(agents: AgentSet, root: str | Path, overwrite: bool = False) -> MemmapManifest:
    root = Path(root)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = MemmapManifest(
        root=root,
        n_agents=agents.n_agents,
        n_resources=agents.n_resources,
        demands_dtype=str(agents.demands.dtype),
        durations_dtype=str(agents.durations.dtype),
    )
    demands = np.memmap(
        manifest.demands_path,
        mode="w+",
        dtype=agents.demands.dtype,
        shape=agents.demands.shape,
    )
    demands[:] = agents.demands
    demands.flush()
    durations = np.memmap(
        manifest.durations_path,
        mode="w+",
        dtype=agents.durations.dtype,
        shape=agents.durations.shape,
    )
    durations[:] = agents.durations
    durations.flush()
    ids = np.memmap(manifest.ids_path, mode="w+", dtype=np.int64, shape=(agents.n_agents,))
    ids[:] = agents.ids
    ids.flush()
    order = np.memmap(
        manifest.arrival_order_path,
        mode="w+",
        dtype=np.int64,
        shape=(agents.n_agents,),
    )
    order[:] = agents.arrival_order
    order.flush()
    metadata = {"manifest": manifest.as_dict(), "agent_metadata": agents.metadata}
    manifest.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest


def load_manifest(root: str | Path) -> MemmapManifest:
    root = Path(root)
    raw = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    data = raw["manifest"]
    return MemmapManifest(
        root=root,
        n_agents=int(data["n_agents"]),
        n_resources=int(data["n_resources"]),
        demands_dtype=str(data["demands_dtype"]),
        durations_dtype=str(data["durations_dtype"]),
    )


def open_memmap(root: str | Path, mode: str = "r") -> AgentSet:
    manifest = load_manifest(root)
    metadata_raw = json.loads(manifest.metadata_path.read_text(encoding="utf-8"))
    demands = np.memmap(
        manifest.demands_path,
        mode=mode,
        dtype=np.dtype(manifest.demands_dtype),
        shape=(manifest.n_agents, manifest.n_resources),
    )
    durations = np.memmap(
        manifest.durations_path,
        mode=mode,
        dtype=np.dtype(manifest.durations_dtype),
        shape=(manifest.n_agents,),
    )
    ids = np.memmap(manifest.ids_path, mode=mode, dtype=np.int64, shape=(manifest.n_agents,))
    order = np.memmap(
        manifest.arrival_order_path,
        mode=mode,
        dtype=np.int64,
        shape=(manifest.n_agents,),
    )
    return AgentSet(
        demands=demands,
        durations=durations,
        ids=ids,
        arrival_order=order,
        metadata=dict(metadata_raw.get("agent_metadata", {})),
    )


class ActiveSet:
    """Mutable membership structure for unscheduled agents.

    A boolean mask gives constant-time deletion and compact memory.  ``indices``
    is vectorised and therefore acceptable for the tens of full passes used by
    million-agent wave scheduling.
    """

    __slots__ = ("_mask", "_remaining")

    def __init__(self, n_agents: int) -> None:
        if n_agents <= 0:
            raise ValidationError("n_agents must be positive")
        self._mask = np.ones(n_agents, dtype=np.bool_)
        self._remaining = n_agents

    @property
    def remaining(self) -> int:
        return self._remaining

    @property
    def empty(self) -> bool:
        return self._remaining == 0

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def contains(self, index: int) -> bool:
        return bool(self._mask[index])

    def remove(self, indices: IntArray | np.ndarray) -> None:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return
        if idx.min(initial=0) < 0 or idx.max(initial=-1) >= self._mask.size:
            raise IndexError("active-set removal contains an out-of-range index")
        unique = np.unique(idx)
        removed = int(np.count_nonzero(self._mask[unique]))
        self._mask[unique] = False
        self._remaining -= removed

    def indices(self) -> IntArray:
        return np.flatnonzero(self._mask).astype(np.int64, copy=False)

    def iter_chunks(self, chunk_size: int) -> Iterator[IntArray]:
        if chunk_size <= 0:
            raise ValidationError("chunk_size must be positive")
        active = self.indices()
        for start in range(0, active.size, chunk_size):
            yield active[start : start + chunk_size]


__all__ = [
    "ActiveSet",
    "MemmapManifest",
    "load_manifest",
    "open_memmap",
    "save_memmap",
]
