from cec2017.optimization.base import Optimizer
from cec2017.optimization.io import (
    RESULT_COLUMNS,
    RESULT_INDEX,
    load_results_table,
    results_table_complete,
    save_boxplots,
    fmin_by_function_dir,
    fmin_log1p_by_function_dir,
    save_fmin_boxplots_per_function,
    save_fmin_log1p_boxplot,
    save_fmin_log1p_boxplots_per_function,
    save_fmin_log1p_plots,
    save_boxplots_per_run,
    save_results_table,
    save_results_tables,
    stage2_scipy_complete,
    stage2_table_complete,
    stage3_deap_complete,
    stage3_pso_complete,
    stage3_pygad_complete,
)
from cec2017.optimization.results import TrialResult
from cec2017.optimization.scipy_minimize import ScipyMinimizeOptimizer

__all__ = [
    "Optimizer",
    "RESULT_COLUMNS",
    "RESULT_INDEX",
    "ScipyMinimizeOptimizer",
    "TrialResult",
    "load_results_table",
    "results_table_complete",
    "save_boxplots",
    "fmin_by_function_dir",
    "fmin_log1p_by_function_dir",
    "save_fmin_boxplots_per_function",
    "save_fmin_log1p_boxplot",
    "save_fmin_log1p_boxplots_per_function",
    "save_fmin_log1p_plots",
    "save_boxplots_per_run",
    "save_results_table",
    "save_results_tables",
    "stage2_scipy_complete",
    "stage2_table_complete",
    "stage3_deap_complete",
    "stage3_pso_complete",
    "stage3_pygad_complete",
]


def __getattr__(name: str):
    """Lazy-load metaheuristic optimizers (heavy optional dependencies)."""
    if name == "DeapGAOptimizer":
        from cec2017.optimization.ga.deap import DeapGAOptimizer

        return DeapGAOptimizer
    if name == "PyGADOptimizer":
        from cec2017.optimization.ga.pygad import PyGADOptimizer

        return PyGADOptimizer
    if name == "PySwarmsPSOOptimizer":
        from cec2017.optimization.pyswarms.pso import PySwarmsPSOOptimizer

        return PySwarmsPSOOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__ + ["DeapGAOptimizer", "PyGADOptimizer", "PySwarmsPSOOptimizer"])
