from pathlib import Path

import pandas as pd

STAGE2_COLUMNS = [f"f{i}" for i in range(1, 29)]
STAGE2_INDEX = [f"run_{i}" for i in range(1, 11)]


def save_results_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return path.resolve()


def load_results_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def stage2_table_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    table = load_results_table(path)
    if table.shape != (10, 28):
        return False
    return list(table.columns) == STAGE2_COLUMNS and list(table.index) == STAGE2_INDEX
