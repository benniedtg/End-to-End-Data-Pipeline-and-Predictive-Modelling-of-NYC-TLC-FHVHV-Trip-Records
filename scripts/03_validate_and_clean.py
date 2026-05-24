"""Read interim samples, write a quality report, clean rows, attach zones, persist."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_INTERIM, OUT_TABLES, ANALYSIS_YEAR
from src.validate import quality_report
from src.clean import clean_trips
from src.zones import load_zone_lookup, attach_zones


def _process(sample_path: Path, out_path: Path, valid_year: int, report_path: Path | None) -> None:
    df = pd.read_parquet(sample_path)
    rep = quality_report(df, valid_year=valid_year)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rep, indent=2, default=str))
        print(f"quality report → {report_path}")
    cleaned = clean_trips(df, valid_year=valid_year)
    zones = load_zone_lookup()
    enriched = attach_zones(cleaned, zones)
    enriched.to_parquet(out_path, index=False)
    print(f"clean+zones → {out_path}  ({len(enriched):,} rows)")


def main() -> None:
    _process(
        sample_path=DATA_INTERIM / f"sample_{ANALYSIS_YEAR}.parquet",
        out_path=DATA_INTERIM / f"clean_{ANALYSIS_YEAR}.parquet",
        valid_year=ANALYSIS_YEAR,
        report_path=OUT_TABLES / f"quality_report_{ANALYSIS_YEAR}.json",
    )
    _process(
        sample_path=DATA_INTERIM / "sample_2022-12_oot.parquet",
        out_path=DATA_INTERIM / "clean_2022-12_oot.parquet",
        valid_year=2022,
        report_path=OUT_TABLES / "quality_report_2022-12_oot.json",
    )


if __name__ == "__main__":
    main()
