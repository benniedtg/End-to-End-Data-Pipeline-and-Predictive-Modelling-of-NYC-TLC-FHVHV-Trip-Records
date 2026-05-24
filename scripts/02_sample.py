"""Build the analysis sample: 5M rows from 2023 (stratified per month) + 200k OOT rows from 2022-12.

Justification (written into the report):
  - Full year 2023 captures all seasonality (summer peaks, winter dips, holidays).
  - Per-month equal quota prevents month-imbalance bias.
  - 5M rows is large enough for robust ML on 20+ features yet fits a single-machine pandas workflow.
  - 2022-12 reserved as an out-of-time (OOT) generalization test, never seen during training.
"""
from pathlib import Path
import pandas as pd

from src.config import (
    DATA_RAW, DATA_INTERIM, ANALYSIS_YEAR, ANALYSIS_MONTHS,
    TARGET_SAMPLE_ROWS, OUT_OF_TIME_FILE, RANDOM_SEED,
)
from src.load import read_sample


def main() -> None:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    per_file = TARGET_SAMPLE_ROWS // len(ANALYSIS_MONTHS)
    frames = []
    for m in ANALYSIS_MONTHS:
        p = DATA_RAW / f"fhvhv_tripdata_{ANALYSIS_YEAR}-{m:02d}.parquet"
        df = read_sample(p, n_rows_target=per_file, seed=RANDOM_SEED + m)
        df["source_file"] = p.name
        frames.append(df)
        print(f"{p.name}: {len(df):,} rows")
    sample = pd.concat(frames, ignore_index=True)
    out = DATA_INTERIM / f"sample_{ANALYSIS_YEAR}.parquet"
    sample.to_parquet(out, index=False)
    print(f"\nTotal 2023 sample: {len(sample):,} rows → {out}")

    oot_p = DATA_RAW / OUT_OF_TIME_FILE
    oot = read_sample(oot_p, n_rows_target=200_000, seed=RANDOM_SEED)
    oot["source_file"] = oot_p.name
    oot_out = DATA_INTERIM / "sample_2022-12_oot.parquet"
    oot.to_parquet(oot_out, index=False)
    print(f"OOT sample: {len(oot):,} rows → {oot_out}")


if __name__ == "__main__":
    main()
