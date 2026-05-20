from time import perf_counter

import numpy as np
import pygad

from cec2017.benchmark.loader import CEC2017Benchmark
from cec2017.optimization.base import Function, Optimizer
from cec2017.optimization.results import TrialResult


class PyGADOptimizer(Optimizer):
    """Genetic algorithm using the PyGAD library.

    Args:
        bounds: Inclusive ``(low, high)`` per dimension.
        sol_per_pop: Population size (solutions per population).
        num_generations: Number of generations.
        num_parents_mating: Parents selected for crossover each generation.
    """

    def __init__(
        self,
        bounds: tuple[float, float] = CEC2017Benchmark.DOMAIN,
        sol_per_pop: int = 50,
        num_generations: int = 100,
        num_parents_mating: int = 25,
    ) -> None:
        self.bounds = bounds
        self.sol_per_pop = sol_per_pop
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating

    def optimize(self, func: Function, dimension: int, seed: int) -> TrialResult:
        """Run one PyGAD GA trial.

        Args:
            func: CEC benchmark function.
            dimension: Problem dimensionality.
            seed: RNG seed passed to PyGAD.

        Returns:
            Trial result with best fitness and elapsed time.
        """
        low, high = self.bounds
        gene_space = [{"low": low, "high": high} for _ in range(dimension)]

        def fitness(ga_instance, batch_solutions, batch_indices) -> np.ndarray:
            solutions = np.asarray(batch_solutions, dtype=float)
            if solutions.ndim == 1:
                solutions = solutions.reshape(1, -1)
            return np.asarray(func(solutions), dtype=float).reshape(-1)

        ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.sol_per_pop,
            num_genes=dimension,
            fitness_func=fitness,
            gene_space=gene_space,
            random_seed=seed,
        )

        start = perf_counter()
        ga.run()
        elapsed_sec = perf_counter() - start
        f_min = float(np.asarray(ga.best_solution()[1]).reshape(-1)[0])
        return TrialResult(f_min=f_min, elapsed_sec=elapsed_sec)
