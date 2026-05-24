"""Train HistGradientBoosting model; evaluate on test + OOT; persist."""
from __future__ import annotations
import json
import joblib
import pandas as pd

from src.config import DATA_PROCESSED, OUT_MODELS
from src.modeling import (
    FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL, TARGET,
    build_gbm_pipeline, evaluate,
)


def main() -> None:
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA_PROCESSED / "train_2023.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test_2023.parquet")
    oot = pd.read_parquet(DATA_PROCESSED / "oot_2022-12.parquet")

    feature_cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL
    pipe = build_gbm_pipeline()
    pipe.fit(train[feature_cols], train[TARGET])

    metrics = {
        "test_2023": evaluate(test[TARGET], pipe.predict(test[feature_cols])),
        "oot_2022_12": evaluate(oot[TARGET], pipe.predict(oot[feature_cols])),
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(pipe, OUT_MODELS / "gbm.joblib")
    (OUT_MODELS / "metrics_gbm.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
