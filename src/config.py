"""Project-wide configuration. Import constants from here; do not hardcode paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data layout
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
ZONE_LOOKUP_PATH = DATA_RAW / "taxi_zone_lookup.csv"

# Outputs
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_MODELS = ROOT / "outputs" / "models"

# Scope
ANALYSIS_YEAR = 2023
ANALYSIS_MONTHS = list(range(1, 13))
OUT_OF_TIME_FILE = "fhvhv_tripdata_2022-12.parquet"  # held-out for OOT test

# Sampling
TARGET_SAMPLE_ROWS = 5_000_000
RANDOM_SEED = 42

# Columns kept from raw parquet (drop originating_base_num — sparse, not useful)
RAW_COLUMNS = [
    "hvfhs_license_num", "request_datetime", "pickup_datetime", "dropoff_datetime",
    "PULocationID", "DOLocationID",
    "trip_miles", "trip_time",
    "base_passenger_fare", "tolls", "bcf", "sales_tax",
    "congestion_surcharge", "airport_fee", "tips", "driver_pay",
    "shared_request_flag", "shared_match_flag",
    "access_a_ride_flag", "wav_request_flag", "wav_match_flag",
]

# Plausibility bounds (filters applied in clean.py — justified in report)
MIN_TRIP_SECONDS = 60          # < 1 min implausible
MAX_TRIP_SECONDS = 4 * 3600    # > 4h implausible for FHVHV
MIN_TRIP_MILES = 0.1
MAX_TRIP_MILES = 100.0
MIN_FARE = 0.0
MAX_FARE = 500.0

# License code → readable name (from TLC FHVHV docs)
LICENSE_CODE_MAP = {
    "HV0002": "Juno",
    "HV0003": "Uber",
    "HV0004": "Via",
    "HV0005": "Lyft",
}
