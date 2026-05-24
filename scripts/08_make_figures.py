"""Render the bokeh figures to outputs/figures/*.html."""
from __future__ import annotations
import joblib
import pandas as pd
from bokeh.io import output_file, save

from src.config import DATA_PROCESSED, OUT_FIGURES, OUT_MODELS
from src.modeling import FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL, TARGET
from src.plots import trips_by_hour, duration_by_dow, monthly_volume, borough_heatmap, residual_scatter


def _save(fig, name: str) -> None:
    path = OUT_FIGURES / f"{name}.html"
    output_file(str(path), title=name)
    save(fig)
    print(f"figure → {path}")


def main() -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(DATA_PROCESSED / "features_2023.parquet")
    _save(trips_by_hour(feats), "01_trips_by_hour")
    _save(duration_by_dow(feats), "02_duration_by_dow")
    _save(monthly_volume(feats), "03_monthly_volume")
    _save(borough_heatmap(feats), "04_borough_heatmap")

    pipe = joblib.load(OUT_MODELS / "gbm.joblib")
    test = pd.read_parquet(DATA_PROCESSED / "test_2023.parquet")
    feat_cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL
    preds = pipe.predict(test[feat_cols])
    _save(residual_scatter(test[TARGET].values, preds), "05_residuals")


if __name__ == "__main__":
    main()
