from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def fmin_log1p_table(table: pd.DataFrame) -> pd.DataFrame:
    """Apply ``log1p`` element-wise to ``f_min`` values (keeps index/columns)."""
    return pd.DataFrame(
        np.log1p(table.to_numpy(dtype=float)),
        index=table.index,
        columns=table.columns,
    )


class ResultsBoxplotter:
    """Boxplots of per-function distributions across independent runs."""

    def __init__(self, width: int = 1200, height: int = 900, png_scale: int = 3) -> None:
        self.width = width
        self.height = height
        self.png_scale = png_scale

    def plot(self, table: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure()
        for column in table.columns:
            fig.add_trace(
                go.Box(
                    y=table[column].values,
                    name=column,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            xaxis_title="Function",
            yaxis_title=title,
            showlegend=False,
        )
        return fig

    def plot_single_function(
        self,
        values: pd.Series,
        *,
        function_name: str,
        title: str,
        y_label: str,
    ) -> go.Figure:
        """Boxplot for one function: distribution over independent runs."""
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=values.values,
                name=function_name,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            xaxis_title="",
            yaxis_title=y_label,
            showlegend=False,
        )
        return fig

    def plot_single_run(
        self,
        row: pd.Series,
        *,
        title: str,
        y_label: str,
    ) -> go.Figure:
        fig = go.Figure()
        for name, value in row.items():
            fig.add_trace(
                go.Box(
                    y=[value],
                    name=name,
                    boxpoints="all",
                )
            )
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            xaxis_title="Function",
            yaxis_title=y_label,
            showlegend=False,
        )
        return fig

    def save_per_function(
        self,
        table: pd.DataFrame,
        output_dir: Path,
        *,
        metric_label: str,
        save_png: bool = True,
    ) -> list[Path]:
        """Write one HTML (+ optional PNG) per function column."""
        output_dir.mkdir(parents=True, exist_ok=True)
        html_paths: list[Path] = []
        for function_name in table.columns:
            title = f"{metric_label} — {function_name}"
            fig = self.plot_single_function(
                table[function_name],
                function_name=function_name,
                title=title,
                y_label=metric_label,
            )
            html_path = output_dir / f"{function_name}.html"
            png_path = output_dir / f"{function_name}.png" if save_png else None
            self.save(fig, html_path, png_path)
            html_paths.append(html_path)
        return html_paths

    def save_per_run(
        self,
        table: pd.DataFrame,
        output_dir: Path,
        *,
        metric_label: str,
        save_png: bool = True,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        html_paths: list[Path] = []
        for run_name, row in table.iterrows():
            title = f"{metric_label} — {run_name}"
            fig = self.plot_single_run(row, title=title, y_label=metric_label)
            html_path = output_dir / f"{run_name}.html"
            png_path = output_dir / f"{run_name}.png" if save_png else None
            self.save(fig, html_path, png_path)
            html_paths.append(html_path)
        return html_paths

    def save(self, fig: go.Figure, path_html: Path, path_png: Path | None = None) -> None:
        path_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path_html), include_plotlyjs="cdn")
        if path_png is not None:
            fig.write_image(
                str(path_png),
                width=self.width,
                height=self.height,
                scale=self.png_scale,
            )
