"""Data-quality summarization. Pure function returning a dict; no side effects, no logging."""
from __future__ import annotations
import pandas as pd


def quality_report(df: pd.DataFrame, valid_year: int | None = None) -> dict:
    """Compute counts and ratios that flag obvious data-quality problems.

    Returns a dict so callers can persist as JSON or render in the report.
    """
    n = len(df)
    rep = {"n_rows": n}

    rep["n_duplicate_rows"] = int(df.duplicated().sum())

    rep["null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}

    if "trip_miles" in df:
        rep["n_negative_trip_miles"] = int((df["trip_miles"] < 0).sum())
    if "trip_time" in df:
        rep["n_zero_or_negative_trip_time"] = int((df["trip_time"] <= 0).sum())

    if "pickup_datetime" in df and "dropoff_datetime" in df:
        bad_order = (df["dropoff_datetime"] < df["pickup_datetime"]).sum()
        rep["n_dropoff_before_pickup"] = int(bad_order)

    if "pickup_datetime" in df:
        years = df["pickup_datetime"].dt.year
        rep["pickup_year_min"] = int(years.min()) if years.notna().any() else None
        rep["pickup_year_max"] = int(years.max()) if years.notna().any() else None
        if valid_year is not None:
            rep["n_pickup_outside_year"] = int(((years != valid_year) & years.notna()).sum())
        else:
            # default: anything outside 2020..2024 is suspicious
            rep["n_pickup_outside_year"] = int(((years < 2020) | (years > 2024)).sum())

    if "PULocationID" in df:
        rep["n_invalid_pu_location"] = int(((df["PULocationID"] < 1) | (df["PULocationID"] > 265)).sum())
    if "DOLocationID" in df:
        rep["n_invalid_do_location"] = int(((df["DOLocationID"] < 1) | (df["DOLocationID"] > 265)).sum())

    return rep
