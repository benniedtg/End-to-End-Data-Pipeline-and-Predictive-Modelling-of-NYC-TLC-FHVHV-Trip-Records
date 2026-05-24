"""Walk data/raw, print schema/row counts per file, write outputs/tables/schema_summary.csv."""
from pathlib import Path
import pandas as pd

from src.config import DATA_RAW, OUT_TABLES
from src.load import inspect_file


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(DATA_RAW.glob("fhvhv_tripdata_*.parquet")):
        info = inspect_file(p)
        rows.append({
            "file": p.name,
            "n_rows": info["n_rows"],
            "n_row_groups": info["n_row_groups"],
            "n_columns": len(info["schema"]),
        })
    df = pd.DataFrame(rows)
    out = OUT_TABLES / "schema_summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nTotal rows across all 24 files: {df['n_rows'].sum():,}")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
