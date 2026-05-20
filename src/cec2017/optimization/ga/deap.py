import random
from time import perf_counter

import numpy as np
from deap import base, creator, tools

from cec2017.benchmark.loader import CEC2017Benchmark
from cec2017.optimization.base import Function, Optimizer
from cec2017.optimization.results import TrialResult


def _ensure_deap_creators() -> None:
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


def _clip_individual(
    individual: list[float],
    bounds: tuple[float, float],
) -> list[float]:
    low, high = bounds
    for i in range(len(individual)):
        if individual[i] < low:
            individual[i] = low
        elif individual[i] > high:
            individual[i] = high
    return individual


class DeapGAOptimizer(Optimizer):
    """Genetic algorithm using the DEAP framework.

    Args:
        bounds: Inclusive ``(low, high)`` per dimension.
        population_size: Number of individuals per generation.
        generations: Number of generations.
        crossover_prob: Probability of crossover between mates.
        mutation_prob: Probability of mutating an offspring.
        mutation_sigma: Gaussian mutation standard deviation.
        mutation_indpb: Per-gene mutation probability inside ``mutGaussian``.
        tournament_size: Tournament selection size.
    """

    def __init__(
        self,
        bounds: tuple[float, float] = CEC2017Benchmark.DOMAIN,
        population_size: int = 50,
        generations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        mutation_sigma: float = 20.0,
        mutation_indpb: float = 0.2,
        tournament_size: int = 3,
    ) -> None:
        self.bounds = bounds
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.mutation_indpb = mutation_indpb
        self.tournament_size = tournament_size

    def _build_toolbox(self, func: Function, dimension: int) -> base.Toolbox:
        _ensure_deap_creators()
        low, high = self.bounds
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, low, high)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=dimension,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual: list[float]) -> tuple[float]:
            x = np.asarray(individual, dtype=float).reshape(1, -1)
            return (float(func(x)[0]),)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0.0,
            sigma=self.mutation_sigma,
            indpb=self.mutation_indpb,
        )
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.tournament_size,
        )
        return toolbox

    def optimize(self, func: Function, dimension: int, seed: int) -> TrialResult:
        """Run one DEAP GA trial.

        Args:
            func: CEC benchmark function.
            dimension: Problem dimensionality.
            seed: RNG seed for Python and NumPy.

        Returns:
            Trial result with best fitness and elapsed time.
        """
        random.seed(seed)
        np.random.seed(seed)
        toolbox = self._build_toolbox(func, dimension)

        start = perf_counter()
        population = toolbox.population(n=self.population_size)
        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        hall_of_fame = tools.HallOfFame(1)
        hall_of_fame.update(population)

        for _ in range(self.generations):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    _clip_individual(child1, self.bounds)
                    _clip_individual(child2, self.bounds)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    _clip_individual(mutant, self.bounds)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            population[:] = offspring
            hall_of_fame.update(population)

        elapsed_sec = perf_counter() - start
        f_min = float(hall_of_fame[0].fitness.values[0])
        return TrialResult(f_min=f_min, elapsed_sec=elapsed_sec)
