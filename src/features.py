"""Feature engineering for FHVHV trips. Operates on a cleaned + zone-merged dataframe."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import LICENSE_CODE_MAP


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Does not drop the originals; the downstream pipeline picks columns."""
    out = df.copy()

    # Target
    out["trip_duration_min"] = out["trip_time"] / 60.0

    # Temporal
    pu = out["pickup_datetime"]
    out["pickup_hour"] = pu.dt.hour.astype("int16")
    out["pickup_dow"] = pu.dt.dayofweek.astype("int8")          # Mon=0, Sun=6
    out["pickup_is_weekend"] = out["pickup_dow"] >= 5
    out["pickup_month"] = pu.dt.month.astype("int8")
    out["pickup_day"] = pu.dt.day.astype("int8")

    # Coarse part-of-day (interpretable bins)
    bins = [-1, 5, 9, 16, 20, 23]
    labels = ["late_night", "morning", "midday", "evening", "night"]
    out["pickup_part_of_day"] = pd.cut(out["pickup_hour"], bins=bins, labels=labels)

    # Wait / dispatch
    if "request_datetime" in out:
        out["request_to_pickup_min"] = (
            (out["pickup_datetime"] - out["request_datetime"]).dt.total_seconds() / 60.0
        )

    # Company
    out["company"] = out["hvfhs_license_num"].map(LICENSE_CODE_MAP).fillna("Other")

    # Fare-derived
    fare_cols = ["base_passenger_fare", "tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee", "tips"]
    out["total_amount"] = out[fare_cols].sum(axis=1)
    out["tip_rate"] = np.where(out["base_passenger_fare"] > 0, out["tips"] / out["base_passenger_fare"], 0.0)

    # Speed
    out["avg_speed_mph"] = np.where(out["trip_duration_min"] > 0,
                                    out["trip_miles"] / (out["trip_duration_min"] / 60.0),
                                    np.nan)

    # Spatial
    out["is_airport"] = out["airport_fee"] > 0
    out["same_borough"] = out["pu_borough"] == out["do_borough"]

    # Boolean flag → 0/1
    for c in ["shared_request_flag", "shared_match_flag", "access_a_ride_flag",
              "wav_request_flag", "wav_match_flag"]:
        if c in out:
            out[c + "_bin"] = out[c].astype(str).str.strip().str.upper().eq("Y").astype("int8")

    return out
