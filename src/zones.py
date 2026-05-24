"""Merge the TLC taxi_zone_lookup.csv onto trips: Borough, Zone, service_zone for PU and DO."""
from __future__ import annotations
import pandas as pd

from src.config import ZONE_LOOKUP_PATH


def load_zone_lookup() -> pd.DataFrame:
    z = pd.read_csv(ZONE_LOOKUP_PATH)
    expected = {"LocationID", "Borough", "Zone", "service_zone"}
    missing = expected - set(z.columns)
    if missing:
        raise ValueError(f"zone lookup missing columns: {missing}")
    return z


def attach_zones(trips: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    cols = ["LocationID", "Borough", "Zone", "service_zone"]
    pu = zones[cols].rename(columns={
        "LocationID": "PULocationID",
        "Borough": "pu_borough", "Zone": "pu_zone", "service_zone": "pu_service_zone",
    })
    do = zones[cols].rename(columns={
        "LocationID": "DOLocationID",
        "Borough": "do_borough", "Zone": "do_zone", "service_zone": "do_service_zone",
    })
    out = trips.merge(pu, on="PULocationID", how="left").merge(do, on="DOLocationID", how="left")
    for c in ["pu_borough", "pu_zone", "pu_service_zone", "do_borough", "do_zone", "do_service_zone"]:
        out[c] = out[c].fillna("Unknown")
    return out
