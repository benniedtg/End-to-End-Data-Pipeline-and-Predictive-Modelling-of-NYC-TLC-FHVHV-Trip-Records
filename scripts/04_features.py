"""Build feature matrix for modeling: full features then 80/20 random split on 2023."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_INTERIM, DATA_PROCESSED, ANALYSIS_YEAR, RANDOM_SEED
from src.features import build_features


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_INTERIM / f"clean_{ANALYSIS_YEAR}.parquet")
    feats = build_features(df)
    out = DATA_PROCESSED / f"features_{ANALYSIS_YEAR}.parquet"
    feats.to_parquet(out, index=False)
    print(f"features → {out}  ({len(feats):,} rows, {feats.shape[1]} cols)")

    train, test = train_test_split(feats, test_size=0.2, random_state=RANDOM_SEED)
    train.to_parquet(DATA_PROCESSED / "train_2023.parquet", index=False)
    test.to_parquet(DATA_PROCESSED / "test_2023.parquet", index=False)
    print(f"train: {len(train):,}  test: {len(test):,}")

    oot = pd.read_parquet(DATA_INTERIM / "clean_2022-12_oot.parquet")
    oot_feats = build_features(oot)
    oot_feats.to_parquet(DATA_PROCESSED / "oot_2022-12.parquet", index=False)
    print(f"oot: {len(oot_feats):,}")


if __name__ == "__main__":
    main()
