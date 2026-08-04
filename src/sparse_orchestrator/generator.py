"""Synthetic workloads with one explicit row per agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import GeneratorConfig
from .model import AgentSet, Provider, ValidationError


@dataclass(frozen=True, slots=True)
class GeneratedWorkload:
    agents: AgentSet
    provider: Provider
    cluster_labels: np.ndarray
    cluster_centres: np.ndarray


def _durations(rng: np.random.Generator, cfg: GeneratorConfig) -> np.ndarray:
    n = cfg.n_agents
    if cfg.duration_mode == "unit":
        return np.full(n, cfg.duration_mean, dtype=cfg.dtype)
    if cfg.duration_mode == "lognormal":
        # Convert requested arithmetic mean approximately to a log-space mean.
        sigma = cfg.duration_sigma
        mu = np.log(cfg.duration_mean) - 0.5 * sigma * sigma
        return rng.lognormal(mu, sigma, size=n).astype(cfg.dtype)
    low = max(cfg.duration_mean * (1.0 - cfg.duration_sigma), 1e-6)
    high = cfg.duration_mean * (1.0 + cfg.duration_sigma)
    return rng.uniform(low, high, size=n).astype(cfg.dtype)


def _cluster_centres(rng: np.random.Generator, cfg: GeneratorConfig) -> np.ndarray:
    centres = rng.uniform(
        cfg.background_low,
        cfg.background_high,
        size=(cfg.n_clusters, cfg.n_resources),
    )
    for cluster in range(cfg.n_clusters):
        dominant = cluster % cfg.n_resources
        centres[cluster, dominant] = rng.uniform(cfg.dominant_low, cfg.dominant_high)
        # A secondary resource makes the profiles nontrivially correlated.
        secondary = (dominant + 1 + cluster // cfg.n_resources) % cfg.n_resources
        centres[cluster, secondary] *= rng.uniform(1.5, 3.0)
    return centres


def _counts(rng: np.random.Generator, cfg: GeneratorConfig) -> np.ndarray:
    alpha = np.full(cfg.n_clusters, cfg.burst_concentration, dtype=np.float64)
    probabilities = rng.dirichlet(alpha)
    return rng.multinomial(cfg.n_agents, probabilities)


def generate_correlated_bursts(cfg: GeneratorConfig, provider: Provider) -> GeneratedWorkload:
    """Generate resource-correlated contiguous arrival bursts.

    Every row is independently perturbed, so agents remain distinct even when
    they belong to the same burst family.  The contiguous arrival order is a
    legitimate stress condition for FCFS: nearby agents tend to compete for the
    same dominant resource, whereas a global scheduler may interleave
    complementary resource profiles.
    """

    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    centres = _cluster_centres(rng, cfg)
    counts = _counts(rng, cfg)
    order = rng.permutation(cfg.n_clusters)
    demands = np.empty((cfg.n_agents, cfg.n_resources), dtype=cfg.dtype)
    labels = np.empty(cfg.n_agents, dtype=np.int16 if cfg.n_clusters < 32768 else np.int32)

    cursor = 0
    for cluster in order:
        count = int(counts[cluster])
        if count == 0:
            continue
        stop = cursor + count
        multiplier = rng.lognormal(
            mean=0.0,
            sigma=cfg.jitter_sigma,
            size=(count, cfg.n_resources),
        )
        additive = rng.uniform(1e-4, 1.0, size=(count, cfg.n_resources))
        demands[cursor:stop] = centres[cluster][None, :] * multiplier + additive
        labels[cursor:stop] = cluster
        cursor = stop

    if cursor != cfg.n_agents:
        raise RuntimeError(f"generator wrote {cursor} rows instead of {cfg.n_agents}")
    if np.any(demands > provider.capacity[None, :]):
        raise ValidationError("generated an agent larger than provider capacity")

    agents = AgentSet(
        demands=demands,
        durations=_durations(rng, cfg),
        ids=np.arange(cfg.n_agents, dtype=np.int64),
        arrival_order=np.arange(cfg.n_agents, dtype=np.int64),
        metadata={
            "generator": "correlated_bursts",
            "seed": cfg.seed,
            "n_clusters": cfg.n_clusters,
            "explicit_agents": True,
        },
    )
    return GeneratedWorkload(agents, provider, labels, centres)


def generate_iid(cfg: GeneratorConfig, provider: Provider) -> GeneratedWorkload:
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    centres = _cluster_centres(rng, cfg)
    probabilities = rng.dirichlet(np.full(cfg.n_clusters, cfg.burst_concentration))
    labels = rng.choice(cfg.n_clusters, size=cfg.n_agents, p=probabilities)
    multiplier = rng.lognormal(
        mean=0.0,
        sigma=cfg.jitter_sigma,
        size=(cfg.n_agents, cfg.n_resources),
    )
    demands = centres[labels] * multiplier + rng.uniform(
        1e-4, 1.0, size=(cfg.n_agents, cfg.n_resources)
    )
    demands = demands.astype(cfg.dtype)
    agents = AgentSet(
        demands=demands,
        durations=_durations(rng, cfg),
        metadata={"generator": "iid", "seed": cfg.seed, "explicit_agents": True},
    )
    return GeneratedWorkload(agents, provider, labels, centres)


def generate_complementary(cfg: GeneratorConfig, provider: Provider) -> GeneratedWorkload:
    """Generate approximately balanced complementary resource families."""

    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    centres = _cluster_centres(rng, cfg)
    labels = np.arange(cfg.n_agents, dtype=np.int64) % cfg.n_clusters
    rng.shuffle(labels)
    multiplier = rng.lognormal(
        mean=0.0,
        sigma=cfg.jitter_sigma,
        size=(cfg.n_agents, cfg.n_resources),
    )
    demands = (centres[labels] * multiplier).astype(cfg.dtype)
    agents = AgentSet(
        demands=demands,
        durations=_durations(rng, cfg),
        metadata={
            "generator": "complementary",
            "seed": cfg.seed,
            "explicit_agents": True,
        },
    )
    return GeneratedWorkload(agents, provider, labels, centres)


def generate(cfg: GeneratorConfig, provider: Provider) -> GeneratedWorkload:
    if cfg.pattern == "correlated_bursts":
        return generate_correlated_bursts(cfg, provider)
    if cfg.pattern == "iid":
        return generate_iid(cfg, provider)
    if cfg.pattern == "complementary":
        return generate_complementary(cfg, provider)
    raise ValidationError(f"unknown generator pattern: {cfg.pattern}")


def distinctness_check(agents: AgentSet, sample: int = 100_000) -> dict[str, float | int | bool]:
    """Check that a representative sample does not collapse into repeated profiles."""

    n = min(sample, agents.n_agents)
    if n <= 0:
        raise ValidationError("sample must be positive")
    idx = np.linspace(0, agents.n_agents - 1, n, dtype=np.int64)
    selected = np.ascontiguousarray(agents.demands[idx])
    rows = selected.view(np.dtype((np.void, selected.dtype.itemsize * selected.shape[1])))
    unique = int(np.unique(rows).size)
    return {
        "sampled_agents": n,
        "unique_sampled_demands": unique,
        "unique_fraction": unique / n,
        "all_sampled_distinct": unique == n,
    }


__all__ = [
    "GeneratedWorkload",
    "distinctness_check",
    "generate",
    "generate_complementary",
    "generate_correlated_bursts",
    "generate_iid",
]
