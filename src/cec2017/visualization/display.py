from pathlib import Path


def display_png_grid(png_dir: Path, *, cols: int = 2, width: int = 450) -> None:
    from IPython.display import HTML, display

    paths = sorted(
        png_dir.glob("f*.png"),
        key=lambda p: int(p.stem.removeprefix("f")),
    )
    for start in range(0, len(paths), cols):
        row_paths = paths[start : start + cols]
        cells = "".join(
            f'<td><img src="{p.as_posix()}" width="{width}" alt="{p.stem}"/></td>'
            for p in row_paths
        )
        if len(row_paths) < cols:
            cells += "<td></td>" * (cols - len(row_paths))
        display(HTML(f"<table><tr>{cells}</tr></table>"))
