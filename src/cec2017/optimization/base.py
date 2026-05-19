from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

Function = Callable[[np.ndarray], np.ndarray]


class Optimizer(ABC):
    """Base class for single-objective optimizers on CEC-2017 functions."""

    @staticmethod
    def wrap_objective(func: Function) -> Callable[[np.ndarray], float]:
        def objective(x: np.ndarray) -> float:
            sample = np.asarray(x, dtype=float).reshape(1, -1)
            return float(func(sample)[0])

        return objective

    @abstractmethod
    def optimize(self, func: Function, dimension: int, seed: int) -> float:
        """Run a single optimization trial.

        Args:
            func: CEC benchmark function.
            dimension: Problem dimensionality.
            seed: RNG seed for the random starting point.

        Returns:
            Best objective value found (``f_min``).
        """
        pass

    def run_table(
        self,
        functions: Sequence[tuple[str, Function]],
        n_runs: int,
        dimension: int,
        base_seed: int = 42,
    ) -> pd.DataFrame:
        """Run multiple trials per function and collect ``f_min`` values.

        Args:
            functions: Sequence of ``(name, callable)`` pairs.
            n_runs: Number of independent runs per function.
            dimension: Problem dimensionality.
            base_seed: Seed for the run-level RNG.

        Returns:
            DataFrame indexed ``run_1`` … ``run_n`` with function names as columns.
        """
        base_rng = np.random.default_rng(base_seed)
        columns = [name for name, _ in functions]
        rows: dict[str, list[float]] = {name: [] for name in columns}

        for _ in range(n_runs):
            for name, func in functions:
                seed = int(base_rng.integers(0, 1_000_000_000))
                rows[name].append(self.optimize(func, dimension, seed))

        data = {name: rows[name] for name in columns}
        index = [f"run_{i}" for i in range(1, n_runs + 1)]
        return pd.DataFrame(data, index=index)
