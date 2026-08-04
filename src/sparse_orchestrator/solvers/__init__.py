"""Mathematically correct reference sparse solvers."""

from .fista import fista_l1
from .irls import irls_lp
from .mp import matching_pursuit
from .ols import orthogonal_least_squares
from .omp import orthogonal_matching_pursuit

__all__ = [
    "fista_l1",
    "irls_lp",
    "matching_pursuit",
    "orthogonal_least_squares",
    "orthogonal_matching_pursuit",
]
