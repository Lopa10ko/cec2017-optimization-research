import importlib
from time import perf_counter

import numpy as np

from cec2017.benchmark.loader import CEC2017Benchmark
from cec2017.optimization.base import Function, Optimizer
from cec2017.optimization.results import TrialResult

ps = importlib.import_module("pyswarms")


class PySwarmsPSOOptimizer(Optimizer):
    """Particle swarm optimization via ``pyswarms.single.GlobalBestPSO``.

    PSO coefficients per run follow the assignment design:
    ``c1 = round(0.5 * U(0,1) + 0.25, 2)``, ``c2 = round(0.3 * U(0,1) + 0.1, 2)``,
    ``w = 0.9``.

    Args:
        bounds: Inclusive ``(low, high)`` per dimension.
        n_particles: Swarm size.
        iters: Number of PSO iterations.
    """

    def __init__(
        self,
        bounds: tuple[float, float] = CEC2017Benchmark.DOMAIN,
        n_particles: int = 10,
        iters: int = 1000,
    ) -> None:
        self.bounds = bounds
        self.n_particles = n_particles
        self.iters = iters

    @staticmethod
    def generate_options(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "c1": float(np.round(0.5 * rng.random() + 0.25, 2)),
            "c2": float(np.round(0.3 * rng.random() + 0.1, 2)),
            "w": 0.9,
        }

    def optimize(self, func: Function, dimension: int, seed: int) -> TrialResult:
        """Run one PSO trial on a CEC function.

        Args:
            func: CEC benchmark function.
            dimension: Problem dimensionality.
            seed: RNG seed for PSO options.

        Returns:
            Trial result; ``f_min`` is the global best cost returned by PySwarms.
        """
        low, high = self.bounds
        options = self.generate_options(seed)
        lower_bounds = np.full(dimension, low, dtype=float)
        upper_bounds = np.full(dimension, high, dtype=float)
        bounds = (lower_bounds, upper_bounds)

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=dimension,
            options=options,
            bounds=bounds,
        )
        objective = self.wrap_batch_objective(func)

        start = perf_counter()
        cost, _position = optimizer.optimize(objective, iters=self.iters)
        elapsed_sec = perf_counter() - start
        return TrialResult(f_min=float(cost), elapsed_sec=elapsed_sec)
