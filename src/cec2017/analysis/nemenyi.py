from dataclasses import dataclass

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


@dataclass(frozen=True)
class NemenyiResult:
    """Outputs of Friedman and Nemenyi tests on benchmark tables."""

    means: pd.DataFrame
    ranks: pd.DataFrame
    average_ranks: pd.Series
    friedman_statistic: float
    friedman_pvalue: float
    pvalues: pd.DataFrame
    critical_difference: float
    alpha: float


class NemenyiComparison:
    """Compare multiple optimizers with Friedman and Nemenyi post-hoc tests.

    Following the blocked design in
    https://www.geeksforgeeks.org/how-to-perform-the-nemenyi-test-in-python/,
    each row is one block (CEC function) and each column is one method. Values
    are mean ``f_min`` over independent runs for that function.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def means_from_tables(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame(
            {name: table.mean(axis=0) for name, table in tables.items()},
            index=tables[next(iter(tables))].columns,
        )

    def ranks_from_means(self, means: pd.DataFrame) -> pd.DataFrame:
        return means.rank(axis=1, method="average")

    def friedman_test(self, means: pd.DataFrame) -> tuple[float, float]:
        columns = [means[name].to_numpy() for name in means.columns]
        statistic, pvalue = stats.friedmanchisquare(*columns)
        return float(statistic), float(pvalue)

    def nemenyi_posthoc(self, means: pd.DataFrame) -> pd.DataFrame:
        matrix = means.to_numpy(dtype=float)
        raw = sp.posthoc_nemenyi_friedman(matrix)
        return pd.DataFrame(
            np.asarray(raw, dtype=float),
            index=list(means.columns),
            columns=list(means.columns),
        )

    def critical_difference(self, n_blocks: int, n_groups: int) -> float:
        q = stats.studentized_range.ppf(1.0 - self.alpha, n_groups, np.inf) / np.sqrt(2.0)
        return float(q * np.sqrt(n_groups * (n_groups + 1) / (6.0 * n_blocks)))

    def run(self, tables: dict[str, pd.DataFrame]) -> NemenyiResult:
        means = self.means_from_tables(tables)
        ranks = self.ranks_from_means(means)
        friedman_statistic, friedman_pvalue = self.friedman_test(means)
        pvalues = self.nemenyi_posthoc(means)
        cd = self.critical_difference(len(means), len(means.columns))
        return NemenyiResult(
            means=means,
            ranks=ranks,
            average_ranks=ranks.mean(axis=0),
            friedman_statistic=friedman_statistic,
            friedman_pvalue=friedman_pvalue,
            pvalues=pvalues,
            critical_difference=cd,
            alpha=self.alpha,
        )
