from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from cec2017.optimization.results import TrialResult

Function = Callable[[np.ndarray], np.ndarray]


class Optimizer(ABC):
    """Base class for single-objective optimizers on CEC-2017 functions."""

    @staticmethod
    def wrap_objective(func: Function) -> Callable[[np.ndarray], float]:
        def objective(x: np.ndarray) -> float:
            sample = np.asarray(x, dtype=float).reshape(1, -1)
            return float(func(sample)[0])

        return objective

    @staticmethod
    def wrap_batch_objective(func: Function) -> Callable[[np.ndarray], np.ndarray]:
        def objective(x: np.ndarray) -> np.ndarray:
            samples = np.asarray(x, dtype=float)
            return np.asarray(func(samples), dtype=float)

        return objective

    @abstractmethod
    def optimize(self, func: Function, dimension: int, seed: int) -> TrialResult:
        """Run a single optimization trial.

        Args:
            func: CEC benchmark function.
            dimension: Problem dimensionality.
            seed: RNG seed for stochastic components.

        Returns:
            Trial result with ``f_min`` and elapsed time.
        """

    def run_tables(
        self,
        functions: Sequence[tuple[str, Function]],
        n_runs: int,
        dimension: int,
        base_seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run multiple trials per function; collect ``f_min`` and runtime tables.

        Args:
            functions: Sequence of ``(name, callable)`` pairs.
            n_runs: Number of independent runs per function.
            dimension: Problem dimensionality.
            base_seed: Seed for the run-level RNG.

        Returns:
            Tuple ``(fmin_table, time_table)`` with index ``run_1`` … ``run_n``.
        """
        base_rng = np.random.default_rng(base_seed)
        columns = [name for name, _ in functions]
        fmin_rows: dict[str, list[float]] = {name: [] for name in columns}
        time_rows: dict[str, list[float]] = {name: [] for name in columns}

        for _ in range(n_runs):
            for name, func in functions:
                seed = int(base_rng.integers(0, 1_000_000_000))
                result = self.optimize(func, dimension, seed)
                fmin_rows[name].append(result.f_min)
                time_rows[name].append(result.elapsed_sec)

        index = [f"run_{i}" for i in range(1, n_runs + 1)]
        fmin_table = pd.DataFrame(fmin_rows, index=index)
        time_table = pd.DataFrame(time_rows, index=index)
        return fmin_table, time_table
