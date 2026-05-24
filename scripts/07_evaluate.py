"""Compare baseline vs GBM side-by-side; compute permutation importance for GBM."""
from __future__ import annotations
import json
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import DATA_PROCESSED, OUT_MODELS, OUT_TABLES, RANDOM_SEED
from src.modeling import FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL, TARGET


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((OUT_MODELS / "metrics_baseline.json").read_text())
    gbm = json.loads((OUT_MODELS / "metrics_gbm.json").read_text())

    rows = []
    for split in ("test_2023", "oot_2022_12"):
        for name, m in (("baseline_ridge", baseline[split]), ("hist_gbm", gbm[split])):
            rows.append({"split": split, "model": name, **m})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT_TABLES / "model_comparison.csv", index=False)
    print(cmp.to_string(index=False))

    # Permutation importance on the test slice (sample down for speed).
    pipe = joblib.load(OUT_MODELS / "gbm.joblib")
    test = pd.read_parquet(DATA_PROCESSED / "test_2023.parquet").sample(50_000, random_state=RANDOM_SEED)
    feature_cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL
    result = permutation_importance(
        pipe, test[feature_cols], test[TARGET],
        n_repeats=3, random_state=RANDOM_SEED, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUT_TABLES / "feature_importance_gbm.csv", index=False)
    print("\nTop features (by perm importance):")
    print(imp.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
