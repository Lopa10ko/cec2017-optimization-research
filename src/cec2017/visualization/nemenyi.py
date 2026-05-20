from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from cec2017.visualization.cd_diagram import draw_cd_diagram, tables_to_perf_df


class NemenyiPlotter:
    """Critical-difference diagram and pairwise p-value heatmap."""

    def __init__(self, width: int = 900, height: int = 420, png_scale: int = 3) -> None:
        self.width = width
        self.height = height
        self.png_scale = png_scale

    def plot_pvalue_heatmap(
        self,
        pvalues: pd.DataFrame,
        *,
        alpha: float,
    ) -> go.Figure:
        """Heatmap of pairwise Nemenyi p-values."""
        labels = list(pvalues.columns)
        z = pvalues.to_numpy(dtype=float)
        text = [[f"{val:.3f}" for val in row] for row in z]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="p-value"),
            )
        )
        fig.update_layout(
            title=f"Nemenyi pairwise p-values (α={alpha:g})",
            width=self.width,
            height=self.height,
            xaxis_title="Method",
            yaxis_title="Method",
        )
        return fig

    def save_plotly(
        self,
        fig: go.Figure,
        path_html: Path,
        path_png: Path | None = None,
    ) -> None:
        path_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path_html), include_plotlyjs="cdn")
        if path_png is not None:
            fig.write_image(
                str(path_png),
                width=self.width,
                height=self.height,
                scale=self.png_scale,
            )

    def save_cd_diagram(
        self,
        tables: dict[str, pd.DataFrame],
        path_html: Path,
        path_png: Path,
        *,
        alpha: float,
    ) -> None:
        """Save CD diagram via `hfawaz/cd-diagram` (Wilcoxon–Holm post-hoc)."""
        import matplotlib.pyplot as plt

        path_png.parent.mkdir(parents=True, exist_ok=True)
        df_perf = tables_to_perf_df(tables)
        fig, _ = draw_cd_diagram(
            df_perf,
            alpha=alpha,
            title=f"Critical difference (Wilcoxon–Holm, α={alpha:g})",
            labels=True,
            filename=path_png,
        )
        plt.close(fig)

        path_html.write_text(
            "\n".join(
                [
                    "<!DOCTYPE html>",
                    "<html><head><meta charset='utf-8'>",
                    "<title>Critical difference diagram</title></head><body>",
                    f"<img src='{path_png.name}' width='{self.width}' "
                    "alt='Critical difference diagram (hfawaz/cd-diagram)'/>",
                    "</body></html>",
                ]
            ),
            encoding="utf-8",
        )

    def save_pvalue_heatmap(
        self,
        pvalues: pd.DataFrame,
        path_html: Path,
        path_png: Path,
        *,
        alpha: float,
    ) -> None:
        fig = self.plot_pvalue_heatmap(pvalues, alpha=alpha)
        self.save_plotly(fig, path_html, path_png)
