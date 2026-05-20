"""Critical difference diagram (hfawaz/cd-diagram, GPL-3.0).

Adapted from https://github.com/hfawaz/cd-diagram — Friedman test, Wilcoxon signed-rank
pairwise comparisons with Holm correction, and Demsar-style CD plot via ``graph_ranks``.
"""

from __future__ import annotations

import math
import operator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Arial"


def tables_to_perf_df(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-format performance table for ``draw_cd_diagram``.

    One row per (method, function) with mean ``f_min`` over runs. Values are negated so
    that higher ``accuracy`` means better optimization (matching hfawaz ranking).
    """
    rows: list[dict[str, object]] = []
    for method, table in tables.items():
        for function in table.columns:
            rows.append(
                {
                    "classifier_name": method,
                    "dataset_name": function,
                    "accuracy": -float(table[function].mean()),
                }
            )
    return pd.DataFrame(rows)


def wilcoxon_holm(
    df_perf: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[list[tuple[str, str, float, bool]], pd.Series, int]:
    """Wilcoxon signed-rank tests with Holm correction (hfawaz/cd-diagram)."""
    classifiers = list(
        df_perf.groupby("classifier_name")
        .size()
        .pipe(lambda s: s[s == s.max()])
        .index
    )
    friedman_pvalue = friedmanchisquare(
        *(
            np.array(
                df_perf.loc[df_perf["classifier_name"] == c]["accuracy"],
                dtype=np.float64,
            )
            for c in classifiers
        )
    )[1]

    p_values: list[tuple[str, str, float, bool]] = []
    m = len(classifiers)
    for i in range(m - 1):
        classifier_1 = classifiers[i]
        perf_1 = np.array(
            df_perf.loc[df_perf["classifier_name"] == classifier_1]["accuracy"],
            dtype=np.float64,
        )
        for j in range(i + 1, m):
            classifier_2 = classifiers[j]
            perf_2 = np.array(
                df_perf.loc[df_perf["classifier_name"] == classifier_2]["accuracy"],
                dtype=np.float64,
            )
            p_value = float(wilcoxon(perf_1, perf_2, zero_method="pratt")[1])
            p_values.append((classifier_1, classifier_2, p_value, False))

    p_values.sort(key=operator.itemgetter(2))
    k = len(p_values)
    for i in range(k):
        new_alpha = alpha / (k - i)
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            break

    sorted_df_perf = df_perf.loc[df_perf["classifier_name"].isin(classifiers)].sort_values(
        ["classifier_name", "dataset_name"]
    )
    max_nb_datasets = int(
        df_perf.groupby("classifier_name").size().max()
    )
    rank_data = np.array(sorted_df_perf["accuracy"]).reshape(m, max_nb_datasets)
    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(classifiers),
        columns=np.unique(sorted_df_perf["dataset_name"]),
    )
    average_ranks = (
        df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    )
    return p_values, average_ranks, max_nb_datasets, float(friedman_pvalue)


def form_cliques(
    p_values: list[tuple[str, str, float, bool]],
    nnames: np.ndarray,
) -> list[list[int]]:
    """Form cliques of methods not significantly different (hfawaz/cd-diagram)."""
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] is False:
            i = int(np.where(nnames == p[0])[0][0])
            j = int(np.where(nnames == p[1])[0][0])
            g_data[min(i, j), max(i, j)] = 1
    graph = networkx.Graph(g_data)
    return list(networkx.find_cliques(graph))


def graph_ranks(
    avranks: np.ndarray | list[float],
    names: list[str],
    p_values: list[tuple[str, str, float, bool]],
    *,
    width: float = 9,
    textspace: float = 1.5,
    reverse: bool = True,
    labels: bool = False,
) -> plt.Figure:
    """Draw a critical difference diagram (hfawaz/cd-diagram / Demsar 2006)."""
    width = float(width)
    textspace = float(textspace)
    sums = list(avranks)
    nnames = np.array(names)
    ssums = sums
    lowv = min(1, int(math.floor(min(ssums))))
    highv = max(len(avranks), int(math.ceil(max(ssums))))
    cline = 0.4
    k = len(ssums)
    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank: float) -> float:
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25
    cline += distanceh
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1.0 / height
    wf = 1.0 / width

    def hfl(values: list[float]) -> list[float]:
        return [a * hf for a in values]

    def wfl(values: list[float]) -> list[float]:
        return [a * wf for a in values]

    def nth(items: list, index: int) -> list:
        return [row[index] for row in items]

    def line(points: list, color: str = "k", **kwargs: object) -> None:
        ax.plot(wfl(nth(points, 0)), hfl(nth(points, 1)), color=color, **kwargs)

    def text(x: float, y: float, s: str, **kwargs: object) -> None:
        ax.text(wf * x, hf * y, s, **kwargs)

    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    for tick_value in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick if tick_value != int(tick_value) else bigtick
        line(
            [(rankpos(tick_value), cline - tick / 2), (rankpos(tick_value), cline)],
            linewidth=2,
        )

    for tick_value in range(lowv, highv + 1):
        text(
            rankpos(tick_value),
            cline - tick / 2 - 0.05,
            str(tick_value),
            ha="center",
            va="bottom",
            size=16,
        )

    space_between_names = 0.24
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line(
            [(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textspace - 0.1, chei)],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + 0.3,
                chei - 0.075,
                format(ssums[i], ".4f"),
                ha="right",
                va="center",
                size=10,
            )
        text(textspace - 0.2, chei, names[i], ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=linewidth,
        )
        if labels:
            text(
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(ssums[i], ".4f"),
                ha="left",
                va="center",
                size=10,
            )
        text(
            textspace + scalewidth + 0.2,
            chei,
            names[i],
            ha="left",
            va="center",
            size=16,
        )

    start = cline + 0.2
    side = -0.02
    bar_height = 0.1
    achieved_half = False
    for clq in form_cliques(p_values, nnames):
        if len(clq) == 1:
            continue
        min_idx = int(np.array(clq).min())
        max_idx = int(np.array(clq).max())
        if min_idx >= len(nnames) / 2 and not achieved_half:
            start = cline + 0.25
            achieved_half = True
        line(
            [
                (rankpos(ssums[min_idx]) - side, start),
                (rankpos(ssums[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
        )
        start += bar_height

    return fig


def draw_cd_diagram(
    df_perf: pd.DataFrame,
    *,
    alpha: float = 0.05,
    title: str | None = None,
    labels: bool = True,
    filename: Path | str | None = None,
) -> tuple[plt.Figure, float]:
    """Friedman + Wilcoxon–Holm and CD diagram; return figure and Friedman p-value."""
    p_values, average_ranks, _, friedman_pvalue = wilcoxon_holm(df_perf, alpha=alpha)
    fig = graph_ranks(
        average_ranks.values,
        list(average_ranks.keys()),
        p_values,
        reverse=True,
        width=9,
        textspace=1.5,
        labels=labels,
    )
    if title:
        fig.suptitle(title, fontsize=18, y=0.98)
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=150)
    return fig, friedman_pvalue
