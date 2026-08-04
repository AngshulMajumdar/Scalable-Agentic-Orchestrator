"""Compare all five reference sparse solvers on one controlled problem."""
from __future__ import annotations

import numpy as np

from sparse_orchestrator.solvers import (
    fista_l1,
    irls_lp,
    matching_pursuit,
    orthogonal_least_squares,
    orthogonal_matching_pursuit,
)

rng = np.random.default_rng(7)
A = rng.normal(size=(16, 64))
A /= np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
truth = np.zeros(64)
truth[[3, 11, 29, 42]] = [1.0, 0.8, 1.2, 0.6]
b = A @ truth

solvers = {
    "MP": lambda: matching_pursuit(A, b, max_atoms=8, positive=True),
    "OMP": lambda: orthogonal_matching_pursuit(A, b, max_atoms=8, positive=True),
    "OLS": lambda: orthogonal_least_squares(A, b, max_atoms=8, positive=True),
    "FISTA": lambda: fista_l1(A, b, l1_lambda=1e-4, max_iterations=1000),
    "IRLS": lambda: irls_lp(A, b, regularization=1e-5, outer_iterations=12),
}

for name, run in solvers.items():
    result = run()
    print(
        f"{name:5s} objective={result.objective:.6e} "
        f"support={result.support.tolist()} iterations={result.iterations}"
    )
