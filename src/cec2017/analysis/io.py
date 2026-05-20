from pathlib import Path

import pandas as pd

from cec2017.analysis.nemenyi import NemenyiResult
from cec2017.optimization.io import load_results_table, results_table_complete

STAGE4_METHOD_PATHS = {
    "scipy": ("stage2", "scipy_fmin.csv"),
    "pso": ("stage3", "pso_fmin.csv"),
    "deap": ("stage3", "deap_fmin.csv"),
    "pygad": ("stage3", "pygad_fmin.csv"),
}

STAGE4_ARTIFACTS = (
    "friedman_summary.csv",
    "nemenyi_means.csv",
    "nemenyi_ranks.csv",
    "nemenyi_average_ranks.csv",
    "nemenyi_pvalues.csv",
    "nemenyi_cd_diagram.html",
    "nemenyi_cd_diagram.png",
    "nemenyi_pvalue_heatmap.html",
    "nemenyi_pvalue_heatmap.png",
)


def stage4_inputs_complete(outputs_root: Path) -> bool:
    """Return True when all Stage 2–3 ``f_min`` tables needed for Stage 4 exist."""
    for stage_dir, filename in STAGE4_METHOD_PATHS.values():
        if not results_table_complete(outputs_root / stage_dir / filename):
            return False
    return True


def load_stage4_fmin_tables(outputs_root: Path) -> dict[str, pd.DataFrame]:
    """Load SciPy, PSO, DEAP, and PyGAD ``f_min`` tables."""
    tables: dict[str, pd.DataFrame] = {}
    for method, (stage_dir, filename) in STAGE4_METHOD_PATHS.items():
        tables[method] = load_results_table(outputs_root / stage_dir / filename)
    return tables


def stage4_complete(output_dir: Path) -> bool:
    """Return True when all Stage 4 artifacts are on disk."""
    if not output_dir.is_dir():
        return False
    return all((output_dir / name).is_file() for name in STAGE4_ARTIFACTS)


def save_nemenyi_results(
    result: NemenyiResult,
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    """Write CSV summaries and figures; return ``output_dir``."""
    from cec2017.visualization.nemenyi import NemenyiPlotter

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "friedman_statistic": [result.friedman_statistic],
            "friedman_pvalue": [result.friedman_pvalue],
            "critical_difference": [result.critical_difference],
            "alpha": [result.alpha],
        }
    ).to_csv(output_dir / "friedman_summary.csv", index=False)

    result.means.to_csv(output_dir / "nemenyi_means.csv")
    result.ranks.to_csv(output_dir / "nemenyi_ranks.csv")
    result.average_ranks.to_frame("average_rank").to_csv(output_dir / "nemenyi_average_ranks.csv")
    result.pvalues.to_csv(output_dir / "nemenyi_pvalues.csv")

    plotter = NemenyiPlotter()
    plotter.save_cd_diagram(
        tables,
        output_dir / "nemenyi_cd_diagram.html",
        output_dir / "nemenyi_cd_diagram.png",
        alpha=result.alpha,
    )
    plotter.save_pvalue_heatmap(
        result.pvalues,
        output_dir / "nemenyi_pvalue_heatmap.html",
        output_dir / "nemenyi_pvalue_heatmap.png",
        alpha=result.alpha,
    )
    return output_dir.resolve()
