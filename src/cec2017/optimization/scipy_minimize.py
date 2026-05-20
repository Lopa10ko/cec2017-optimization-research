from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from cec2017.benchmark.loader import CEC2017Benchmark
from cec2017.optimization.base import Function, Optimizer
from cec2017.optimization.results import TrialResult


class ScipyMinimizeOptimizer(Optimizer):
    """Bound-constrained optimization via ``scipy.optimize.minimize``.

    The ``method`` argument is omitted so SciPy chooses automatically. With
    ``bounds`` supplied, the default is one of ``BFGS``, ``L-BFGS-B``, or
    ``SLSQP`` depending on constraints (see SciPy ``minimize`` documentation).

    Args:
        bounds: Inclusive ``(low, high)`` per dimension.
        maxiter: Maximum iterations per run (passed to the chosen solver).
    """

    def __init__(
        self,
        bounds: tuple[float, float] = CEC2017Benchmark.DOMAIN,
        maxiter: int = 1000,
    ) -> None:
        self.bounds = bounds
        self.maxiter = maxiter

    def optimize(self, func: Function, dimension: int, seed: int) -> TrialResult:
        """Run one SciPy minimization trial with a uniform random start.

        Args:
            func: CEC benchmark function.
            dimension: Problem dimensionality.
            seed: RNG seed for the initial point.

        Returns:
            Trial result with ``f_min`` and wall-clock time.
        """
        rng = np.random.default_rng(seed)
        low, high = self.bounds
        x0 = rng.uniform(low, high, size=dimension)
        scipy_bounds: list[tuple[float, float]] = [self.bounds] * dimension

        start = perf_counter()
        result = minimize(
            fun=self.wrap_objective(func),
            x0=x0,
            bounds=scipy_bounds,
            options={"maxiter": self.maxiter},
        )
        elapsed_sec = perf_counter() - start
        return TrialResult(f_min=float(result.fun), elapsed_sec=elapsed_sec)
