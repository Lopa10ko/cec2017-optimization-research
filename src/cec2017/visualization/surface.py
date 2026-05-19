from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

Function = Callable[[np.ndarray], np.ndarray]
Grid = tuple[np.ndarray, np.ndarray, np.ndarray]


class SurfaceGridBuilder:
    """Builds 2D evaluation grids for 3D surface visualization.

    For dimension > 2, remaining coordinates are fixed at zero (slice at the
    origin in higher dimensions), matching the upstream CEC-2017 surface plot
    convention.
    """

    def __init__(self, domain: tuple[float, float] = (-100.0, 100.0), points: int = 50, dimension: int = 2):
        self.domain = domain
        self.points = points
        self.dimension = dimension

    def build(self, function: Function) -> Grid:
        """Evaluate ``function`` on a regular 2D mesh.

        Args:
            function: Benchmark callable expecting shape ``(m, D)``.

        Returns:
            Tuple ``(X, Y, Z)`` of 2D arrays suitable for surface plotting.
        """
        lo, hi = self.domain
        axis = np.linspace(lo, hi, self.points)
        xys = np.transpose([np.tile(axis, self.points), np.repeat(axis, self.points)])

        if self.dimension > 2:
            tail = np.zeros((xys.shape[0], self.dimension - 2))
            samples = np.concatenate([xys, tail], axis=1)
        else:
            samples = xys

        z_flat = function(samples)
        z = z_flat.reshape(self.points, self.points)
        x = xys[:, 0].reshape(self.points, self.points)
        y = xys[:, 1].reshape(self.points, self.points)
        return x, y, z


class PlotlySurfacePlotter:
    """Renders and exports CEC-2017 function surfaces with Plotly."""

    def __init__(self, png_width: int = 1200, png_height: int = 900, png_scale: int = 3):
        self.png_width = png_width
        self.png_height = png_height
        self.png_scale = png_scale

    def plot_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str) -> go.Figure:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, name=title)])
        fig.update_layout(
            title=title,
            width=self.png_width,
            height=self.png_height,
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
        )
        return fig

    def plot_interactive_gallery(self, grids: dict[str, Grid]) -> go.Figure:
        names = list(grids.keys())
        fig = go.Figure()
        for i, name in enumerate(names):
            x, y, z = grids[name]
            fig.add_trace(
                go.Surface(x=x, y=y, z=z, name=name, visible=(i == 0))
            )

        buttons = []
        for name in names:
            visible = [n == name for n in names]
            buttons.append(
                dict(
                    label=name,
                    method="update",
                    args=[{"visible": visible}, {"title": name}],
                )
            )

        fig.update_layout(
            title=names[0],
            width=self.png_width,
            height=self.png_height,
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.02,
                    y=0.98,
                    buttons=buttons,
                )
            ],
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
        )
        return fig

    def save_png(self, fig: go.Figure, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), scale=self.png_scale)

    def save_html(self, fig: go.Figure, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn")

    def plot_all(
        self,
        functions: Sequence[tuple[str, Function]],
        output_dir: Path,
        points: int = 50,
        dimension: int = 2,
        domain: tuple[float, float] = (-100.0, 100.0),
    ) -> dict[str, Grid]:
        """Build grids, export per-function PNGs and one dropdown HTML gallery.

        Args:
            functions: Sequence of ``(name, callable)`` pairs to plot.
            output_dir: Directory for ``f*.png`` and ``cec_f1_f10.html``.
            points: Grid resolution per axis.
            dimension: Benchmark input dimension.
            domain: Domain bounds for the surface slice.

        Returns:
            Mapping of function names to computed grids.
        """
        builder = SurfaceGridBuilder(domain=domain, points=points, dimension=dimension)
        grids: dict[str, Grid] = {}

        for name, func in functions:
            grids[name] = builder.build(func)
            fig = self.plot_surface(*grids[name], title=name)
            self.save_png(fig, output_dir / f"{name}.png")

        gallery = self.plot_interactive_gallery(grids)
        self.save_html(gallery, output_dir / "cec_f1_f10.html")
        return grids
