from __future__ import annotations

import numpy as np

from sparse_orchestrator.packing import best_fit_refinement, first_fit_window, greedy_pack


def test_greedy_pack_respects_capacity() -> None:
    demands = np.array([[6.0, 1.0], [1.0, 6.0], [5.0, 5.0], [2.0, 2.0]])
    capacity = np.array([10.0, 10.0])
    result = greedy_pack(demands, np.array([0, 1, 2, 3]), capacity, use_numba=False)
    used = demands[result.selected_positions].sum(axis=0)
    assert np.all(used <= capacity + 1e-10)
    np.testing.assert_allclose(result.used, used)


def test_strict_fifo_stops_at_first_failure() -> None:
    demands = np.array([[8.0, 1.0], [3.0, 3.0], [1.0, 8.0]])
    result = first_fit_window(
        demands,
        np.array([0, 1, 2]),
        np.array([10.0, 10.0]),
        stop_at_first_failure=True,
    )
    np.testing.assert_array_equal(result.selected_positions, np.array([0]))
    assert result.considered == 2


def test_windowed_fifo_skips_unfit_agent() -> None:
    demands = np.array([[8.0, 1.0], [3.0, 3.0], [1.0, 8.0]])
    result = first_fit_window(
        demands,
        np.array([0, 1, 2]),
        np.array([10.0, 10.0]),
        stop_at_first_failure=False,
    )
    np.testing.assert_array_equal(result.selected_positions, np.array([0, 2]))


def test_best_fit_refinement_can_use_complementarity() -> None:
    demands = np.array([[8.0, 1.0], [7.0, 2.0], [1.0, 8.0], [2.0, 7.0]])
    result = best_fit_refinement(
        demands,
        np.array([0, 1, 2, 3]),
        np.array([10.0, 10.0]),
        passes=2,
    )
    used = demands[result.selected_positions].sum(axis=0)
    assert np.all(used <= np.array([10.0, 10.0]) + 1e-10)
    assert result.selected_positions.size >= 2
