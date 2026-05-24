"""Parquet schema inspection and row-sampled reading."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import RAW_COLUMNS, RANDOM_SEED


def inspect_file(path: Path) -> dict:
    """Return schema + row count + null counts for one parquet file (cheap; reads metadata only)."""
    pf = pq.ParquetFile(path)
    schema = {f.name: str(f.physical_type) for f in pf.schema}
    n_rows = pf.metadata.num_rows
    return {"path": str(path), "n_rows": n_rows, "n_row_groups": pf.metadata.num_row_groups, "schema": schema}


def read_sample(path: Path, n_rows_target: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Read ~n_rows_target rows from one parquet file by row-group then row-level random subsampling.

    Strategy: read all row groups for the chosen columns, then sample without replacement.
    For ~20M-row files at 24 columns, the full-column read fits in <2GB RAM, so this is fine.
    If memory is a concern on smaller hardware, switch to per-row-group iteration.
    """
    pf = pq.ParquetFile(path)
    df = pf.read(columns=RAW_COLUMNS).to_pandas()
    if len(df) <= n_rows_target:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n_rows_target, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)
