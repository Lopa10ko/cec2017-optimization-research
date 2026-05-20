from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [f"f{i}" for i in range(1, 29)]
RESULT_INDEX = [f"run_{i}" for i in range(1, 11)]

STAGE2_COLUMNS = RESULT_COLUMNS
STAGE2_INDEX = RESULT_INDEX


def save_results_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return path.resolve()


def load_results_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def results_table_complete(path: Path) -> bool:
    """Return True if the CSV exists with shape (10, 28) and expected labels."""
    if not path.is_file():
        return False
    table = load_results_table(path)
    if table.shape != (10, 28):
        return False
    return list(table.columns) == RESULT_COLUMNS and list(table.index) == RESULT_INDEX


def stage2_table_complete(path: Path) -> bool:
    """Alias for a single Stage 2 f_min table (backward compatibility)."""
    return results_table_complete(path)


def stage2_scipy_complete(fmin_path: Path, time_path: Path) -> bool:
    """Return True when both Stage 2 SciPy CSVs are present and valid."""
    return results_table_complete(fmin_path) and results_table_complete(time_path)


def stage3_pso_complete(fmin_path: Path, time_path: Path) -> bool:
    """Return True when both Stage 3 PSO CSVs are present and valid."""
    return results_table_complete(fmin_path) and results_table_complete(time_path)


def stage3_deap_complete(fmin_path: Path, time_path: Path) -> bool:
    """Return True when both Stage 3 DEAP CSVs are present and valid."""
    return results_table_complete(fmin_path) and results_table_complete(time_path)


def stage3_pygad_complete(fmin_path: Path, time_path: Path) -> bool:
    """Return True when both Stage 3 PyGAD CSVs are present and valid."""
    return results_table_complete(fmin_path) and results_table_complete(time_path)


def save_results_tables(
    fmin_df: pd.DataFrame,
    time_df: pd.DataFrame,
    fmin_path: Path,
    time_path: Path,
) -> tuple[Path, Path]:
    return (
        save_results_table(fmin_df, fmin_path),
        save_results_table(time_df, time_path),
    )


def save_boxplots(
    fmin_df: pd.DataFrame,
    time_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    save_png: bool = True,
) -> tuple[Path, Path]:
    """Export f_min and runtime boxplots for one optimizer prefix.

    Args:
        fmin_df: Best objective values per run and function.
        time_df: Runtimes (seconds) per run and function.
        output_dir: Directory for HTML (and optional PNG) artifacts.
        prefix: Filename stem (e.g. ``scipy``, ``pso``, ``deap``).

    Returns:
        Paths to the f_min and time HTML boxplots.
    """
    from cec2017.visualization.boxplot import ResultsBoxplotter

    plotter = ResultsBoxplotter()
    output_dir.mkdir(parents=True, exist_ok=True)

    fmin_html = output_dir / f"{prefix}_fmin_boxplot.html"
    time_html = output_dir / f"{prefix}_time_boxplot.html"
    fmin_fig = plotter.plot(fmin_df, title=f"{prefix}: f_min across runs")
    time_fig = plotter.plot(time_df, title=f"{prefix}: runtime (s) across runs")

    if save_png:
        plotter.save(fmin_fig, fmin_html, output_dir / f"{prefix}_fmin_boxplot.png")
        plotter.save(time_fig, time_html, output_dir / f"{prefix}_time_boxplot.png")
    else:
        plotter.save(fmin_fig, fmin_html)
        plotter.save(time_fig, time_html)

    return fmin_html, time_html


def save_boxplots_per_run(
    fmin_df: pd.DataFrame,
    time_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    save_png: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Export separate boxplots for each run (one figure per run).

    Unlike :func:`save_boxplots`, this does not pool runs into one figure.
    Each run gets its own plot with one box per function (f1–f28).

    Output layout::

        {output_dir}/{prefix}_fmin_by_run/run_1.html
        {output_dir}/{prefix}_fmin_by_run/run_1.png
        ...
        {output_dir}/{prefix}_time_by_run/run_10.png

    Args:
        fmin_df: Best objective values per run and function.
        time_df: Runtimes (seconds) per run and function.
        output_dir: Parent directory for ``*_by_run`` subfolders.
        prefix: Filename stem (e.g. ``pso``).
        save_png: Whether to export high-DPI PNG alongside HTML.

    Returns:
        ``(fmin_html_paths, time_html_paths)`` in run order.
    """
    from cec2017.visualization.boxplot import ResultsBoxplotter

    plotter = ResultsBoxplotter()
    fmin_dir = output_dir / f"{prefix}_fmin_by_run"
    time_dir = output_dir / f"{prefix}_time_by_run"
    fmin_paths = plotter.save_per_run(
        fmin_df,
        fmin_dir,
        metric_label=f"{prefix}: f_min",
        save_png=save_png,
    )
    time_paths = plotter.save_per_run(
        time_df,
        time_dir,
        metric_label=f"{prefix}: runtime (s)",
        save_png=save_png,
    )
    return fmin_paths, time_paths


def fmin_by_function_dir(output_dir: Path, prefix: str) -> Path:
    """Directory for per-function ``f_min`` boxplots of one optimizer."""
    return output_dir / f"{prefix}_fmin_by_function"


def fmin_log1p_by_function_dir(output_dir: Path, prefix: str) -> Path:
    """Directory for per-function ``log1p(f_min)`` boxplots."""
    return output_dir / f"{prefix}_fmin_log1p_by_function"


def save_fmin_log1p_boxplot(
    fmin_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    save_png: bool = True,
) -> Path:
    """Export aggregate boxplot of ``log1p(f_min)`` across functions and runs."""
    from cec2017.visualization.boxplot import ResultsBoxplotter, fmin_log1p_table

    plotter = ResultsBoxplotter()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{prefix}_fmin_log1p_boxplot.html"
    png_path = output_dir / f"{prefix}_fmin_log1p_boxplot.png" if save_png else None
    fig = plotter.plot(
        fmin_log1p_table(fmin_df),
        title=f"{prefix}: log1p(f_min) across runs",
    )
    fig.update_layout(yaxis_title="log1p(f_min)")
    plotter.save(fig, html_path, png_path)
    return html_path


def save_fmin_log1p_boxplots_per_function(
    fmin_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    save_png: bool = True,
) -> list[Path]:
    """Export one ``log1p(f_min)`` boxplot per function (f1–f28)."""
    from cec2017.visualization.boxplot import ResultsBoxplotter, fmin_log1p_table

    plotter = ResultsBoxplotter()
    return plotter.save_per_function(
        fmin_log1p_table(fmin_df),
        fmin_log1p_by_function_dir(output_dir, prefix),
        metric_label=f"{prefix}: log1p(f_min)",
        save_png=save_png,
    )


def save_fmin_log1p_plots(
    fmin_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    save_png: bool = True,
) -> tuple[Path, list[Path]]:
    """Write aggregate and per-function ``log1p(f_min)`` boxplots."""
    aggregate = save_fmin_log1p_boxplot(fmin_df, output_dir, prefix, save_png=save_png)
    per_function = save_fmin_log1p_boxplots_per_function(
        fmin_df, output_dir, prefix, save_png=save_png
    )
    return aggregate, per_function


def save_fmin_boxplots_per_function(
    fmin_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    save_png: bool = True,
) -> list[Path]:
    """Export one ``f_min`` boxplot per function (f1–f28), separate figures.

    Each figure shows the distribution over independent runs for that function only.
    Runtime is not plotted here; use :func:`save_boxplots` for aggregate time charts.

    Output layout::

        {output_dir}/{prefix}_fmin_by_function/f1.html
        {output_dir}/{prefix}_fmin_by_function/f1.png
        ...

    Args:
        fmin_df: Best objective values per run and function.
        output_dir: Parent directory for the ``*_fmin_by_function`` subfolder.
        prefix: Filename stem (e.g. ``pso``, ``deap``, ``pygad``).
        save_png: Whether to export high-DPI PNG alongside HTML.

    Returns:
        HTML paths in function column order.
    """
    from cec2017.visualization.boxplot import ResultsBoxplotter

    plotter = ResultsBoxplotter()
    return plotter.save_per_function(
        fmin_df,
        fmin_by_function_dir(output_dir, prefix),
        metric_label=f"{prefix}: f_min",
        save_png=save_png,
    )
