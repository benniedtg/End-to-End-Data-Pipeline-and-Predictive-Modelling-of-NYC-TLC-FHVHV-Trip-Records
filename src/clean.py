"""Apply plausibility filters and dedupe. Filters are documented in src/config.py."""
from __future__ import annotations
import pandas as pd

from src.config import (
    MIN_TRIP_SECONDS, MAX_TRIP_SECONDS, MIN_TRIP_MILES, MAX_TRIP_MILES,
    MIN_FARE, MAX_FARE,
)


def clean_trips(df: pd.DataFrame, valid_year: int) -> pd.DataFrame:
    """Return a copy of `df` with implausible / duplicate rows removed.

    Order matters: drop nulls in critical columns first (cheapest), then range filters, then dedupe.
    The function returns the same column set it received.
    """
    before = len(df)
    critical = ["pickup_datetime", "dropoff_datetime", "trip_time", "trip_miles", "PULocationID", "DOLocationID"]
    df = df.dropna(subset=[c for c in critical if c in df.columns])

    if "pickup_datetime" in df:
        df = df[df["pickup_datetime"].dt.year == valid_year]
    if "dropoff_datetime" in df and "pickup_datetime" in df:
        df = df[df["dropoff_datetime"] > df["pickup_datetime"]]

    df = df[(df["trip_time"] >= MIN_TRIP_SECONDS) & (df["trip_time"] <= MAX_TRIP_SECONDS)]
    df = df[(df["trip_miles"] >= MIN_TRIP_MILES) & (df["trip_miles"] <= MAX_TRIP_MILES)]

    if "base_passenger_fare" in df:
        df = df[(df["base_passenger_fare"] >= MIN_FARE) & (df["base_passenger_fare"] <= MAX_FARE)]

    df = df[(df["PULocationID"].between(1, 265)) & (df["DOLocationID"].between(1, 265))]

    df = df.drop_duplicates()
    print(f"clean_trips: kept {len(df):,} / {before:,} rows ({len(df)/before:.1%})")
    return df.reset_index(drop=True)
