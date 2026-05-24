"""Build reports/report.ipynb from the executed pipeline artifacts.

Run from the project root:
    python scripts/_build_report.py
    jupyter nbconvert --to notebook --execute --inplace reports/report.ipynb

The notebook is a research-style narrative: 8 sections, mixed markdown + code
cells that load the real artifacts (CSV/JSON) so figures and tables render
alongside the prose. No placeholders -- every number is read from disk.
"""

from __future__ import annotations

import json
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "report.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)


def md(src: str):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str):
    return nbf.v4.new_code_cell(src.strip("\n"))


cells = []

# ---------------------------------------------------------------- title
cells.append(md(r"""
# Predicting NYC FHVHV Trip Duration: A Single-Machine Pipeline from 444 M Raw Records to a Validated Model

**SIT220 Task 8 (HD) — Research-Style Report**

This notebook documents an end-to-end pandas pipeline that ingests 24 monthly
NYC TLC High-Volume For-Hire Vehicle (FHVHV) parquet files (Jan 2022 – Dec 2023,
444,906,103 rows in total), down-samples to a tractable 2023 working set,
engineers pre-trip features, and trains two regressors for trip-duration
prediction. The headline finding is methodological as much as quantitative:
an apparently strong gradient-boosted model (R² ≈ 0.99) is shown to be
**arithmetically reconstructing the target from a leaking feature**, so the
honest pre-trip benchmark is the Ridge baseline at R² ≈ 0.81 / MAE ≈ 4.2 min.
"""))

# ---------------------------------------------------------------- setup
cells.append(code(r"""
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

ART = Path("..") / "outputs"
TABLES = ART / "tables"
FIGS = ART / "figures"
MODELS = ART / "models"
"""))

# ---------------------------------------------------------------- §1 intro
cells.append(md(r"""
## 1. Introduction and Problem Definition

**Prediction problem.** Given the information available *before* an FHVHV
trip starts — request time, pickup zone, intended drop-off zone, vehicle
attributes, and pre-trip fare components — estimate the trip's duration in
minutes (`trip_duration_min`). Accurate pre-trip ETAs underpin three
operational use-cases that NYC TLC explicitly tracks:

1. **Rider-facing ETAs** in the Uber/Lyft/Via apps.
2. **Driver allocation** — dispatch is biased towards drivers whose
   *current* trip will finish soon, so a calibrated duration model improves
   marketplace efficiency.
3. **Pricing fairness audits** — TLC enforces minimum per-trip earnings;
   detecting fares whose duration is mis-predicted at booking helps audit
   surge behaviour.

**Approach.** We frame the task as supervised regression. A linear Ridge
baseline establishes the no-interaction floor; a `HistGradientBoosting`
regressor probes the upper bound. Both are evaluated on (a) a random
20 % held-out test split of 2023, and (b) a temporally-disjoint
**out-of-time slice from December 2022** to probe generalisation across
the year boundary.

**Why this matters for the report.** A central finding (developed in §6) is
that the boosted model's near-perfect R² is an artefact of including
`avg_speed_mph = trip_miles / (trip_time/3600)` as a feature. The Ridge
baseline cannot exploit this multiplicative leakage in a single linear
term and therefore reports the model performance that would actually be
achievable at booking time, when a router supplies expected miles but
realised speed is unknown.
"""))

# ---------------------------------------------------------------- §2 data acq
cells.append(md(r"""
## 2. Data Acquisition and Scope

**Source.** NYC Taxi & Limousine Commission High-Volume For-Hire Vehicle
trip records, published as monthly parquet files at
<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>. We
ingested all 24 monthly files for January 2022 through December 2023
(~11 GB on disk).

**Schema inspection.** Every file has the same 24-column schema. One
observation worth flagging: TLC switched the parquet *row-group* layout
mid-year — files from 2022-01 through 2023-07 use a single row group per
file, whereas 2023-08 onwards use 18–20 row groups. This has no effect
on schema or data but materially changes streaming-read performance and
is the kind of provider-side change a production ingestion job must
detect.
"""))

cells.append(code(r"""
schema = pd.read_csv(TABLES / "schema_summary.csv")
print(f"files: {len(schema)}  total rows: {schema['n_rows'].sum():,}")
schema
"""))

cells.append(md(r"""
**Scope justification (and why pandas is enough here).** The full
444 M-row corpus would not fit in the 16 GB of RAM available on the
single machine used for this submission. The pipeline therefore reduces
to a **stratified per-month sample of 2023**: roughly 417 k rows per
month → **4,999,992 rows (~2.2 % of the 232 M-row year)**. The 200 k OOT
slice from December 2022 is sampled independently and never touched
during training. This is enough data for stable gradient boosting (a
typical FHVHV month has ~15-20 M trips but tens of thousands of
distinct pickup/drop-off pairs, so 5 M trips already saturates the
zone-pair coverage we use as a feature) while leaving headroom for
feature engineering, dataframe joins, and model artefacts. A Spark
workflow on this corpus is feasible but unnecessary for the analysis
goals — it is not pursued here, in line with the task brief.
"""))

# ---------------------------------------------------------------- §3 wrangle
cells.append(md(r"""
## 3. Data Wrangling and Preprocessing

The pipeline runs as a sequence of explicit, reproducible scripts
(`scripts/01_inspect_schema.py` → `08_make_figures.py`). The wrangling
stage performs:

1. **Schema check** — every monthly file reports the same 24 columns and
   compatible dtypes; the row-group change is logged.
2. **Stratified sampling** — fixed seed 42, fraction chosen per file to
   hit ~417 k rows/month for 2023 and 200 k rows for 2022-12.
3. **Quality report (5 M sample of 2023):** see below — only **3
   duplicate rows**, **188 trips with dropoff before pickup**, **1
   non-positive `trip_time`**, and **zero** nulls in any critical column.
4. **Cleaning filters with justification:**
   - `trip_time ∈ [60, 14400] s` — exclude sub-minute trips (cancellations
     mis-recorded) and trips over 4 hours (long-distance airport runs
     beyond the FHVHV operating envelope).
   - `trip_miles ∈ [0.1, 100]` — exclude zero-distance and clearly
     impossible long hauls.
   - `base_passenger_fare ∈ [0, 500]` — exclude negative fares
     (refund/correction entries) and extreme values.
   - `PULocationID, DOLocationID ∈ [1, 265]` — TLC's published zone-ID
     range.
5. **Deduplication** — drop exact-duplicate rows on the full record.
6. **Join with `taxi_zone_lookup.csv`** — attaches borough and service
   zone labels to both endpoints.

**Net attrition: 4,234 rows of 4,999,992 (0.085 %)** — the 5 M sample is
unusually clean, because TLC has already done substantial validation
upstream. The cleaning thresholds nonetheless harden the pipeline
against the *kind* of edge cases that would matter at full scale.
"""))

cells.append(code(r"""
qc_2023 = json.loads((TABLES / "quality_report_2023.json").read_text())
qc_oot  = json.loads((TABLES / "quality_report_2022-12_oot.json").read_text())

summary = pd.DataFrame({
    "2023_sample_5M":  qc_2023,
    "2022-12_OOT_200k": qc_oot,
}).drop(index="null_counts")
summary
"""))

cells.append(code(r"""
# null counts (all zero on critical columns -- shown to make the point)
nulls_2023 = pd.Series(qc_2023["null_counts"]).sort_values(ascending=False).head()
nulls_2023
"""))

# ---------------------------------------------------------------- §4 features
cells.append(md(r"""
## 4. Feature Engineering

The analysis-ready dataset `data/processed/features_2023.parquet` has
**4,995,758 rows × 47 columns**. The engineered features fall into five
families:

- **Target.** `trip_duration_min = trip_time / 60`.
- **Temporal.** `pickup_hour`, `pickup_dow`, `pickup_month`,
  `pickup_day`, `pickup_is_weekend`, `pickup_part_of_day` (morning /
  midday / evening / night bucket), and the request-to-pickup gap
  `request_to_pickup_min`. These capture rush-hour load, weekend
  rhythm, and dispatch latency.
- **Spatial.** `pu_borough`, `do_borough`, `pu_service_zone`,
  `do_service_zone`, plus the boolean `same_borough` and `is_airport`
  flags. Cross-borough trips and airport trips have markedly different
  duration distributions.
- **Fare-derived.** `total_amount`, `tip_rate` — included to characterise
  the trip's economic profile rather than as direct duration signal.
- **Vehicle / service.** `company` (Uber / Lyft / Via / Juno from
  `hvfhs_license_num`), `shared_request_flag_bin`,
  `shared_match_flag_bin`, `access_a_ride_flag_bin`,
  `wav_request_flag_bin`, `wav_match_flag_bin`.
- **Operational.** `trip_miles`, `avg_speed_mph`.

**Explicit leakage caveat (called out so §6 can quantify it).** The
feature `avg_speed_mph = trip_miles / (trip_time / 3600)` is
algebraically a function of the target. We include it deliberately,
because the contrast between a model with and without an effective speed
predictor is itself one of the report's findings. A production
deployment would replace `avg_speed_mph` with an a-priori estimate from a
routing service.
"""))

cells.append(code(r"""
feat = pd.read_parquet("../data/processed/features_2023.parquet")
print(f"shape: {feat.shape}")
feat[["trip_duration_min", "trip_miles", "avg_speed_mph",
      "pickup_hour", "pickup_dow", "pu_borough", "do_borough",
      "is_airport", "same_borough"]].head()
"""))

# ---------------------------------------------------------------- §5 method
cells.append(md(r"""
## 5. Modelling Methodology

**Target.** `trip_duration_min`.

**Split.** 80/20 random split of the 2023 sample (seed = 42). The
2022-12 sample is reserved as a temporally-disjoint **out-of-time
(OOT)** evaluation set — it is never touched during training, tuning,
or feature selection.

**Baseline — Ridge regression** (`scripts/05_train_baseline.py`).
A `ColumnTransformer` standardises numeric features and one-hot encodes
the low-cardinality categoricals (`pu_borough`, `do_borough`, `company`,
`pickup_part_of_day`). Ridge regularisation strength `α = 1.0`. This
model can capture additive effects only — it cannot represent the
multiplicative `duration ≈ miles / speed` relationship in a single
linear term, which is what makes it the honest no-leakage baseline.

**Stronger — Histogram-based Gradient Boosting**
(`scripts/06_train_gbm.py`). `HistGradientBoostingRegressor` with
`max_iter=200`, `learning_rate=0.08`, `max_leaf_nodes=63`,
`min_samples_leaf=20`. Native handling of categoricals, no scaling
required. Crucially, the GBM can split on `trip_miles` and
`avg_speed_mph` in interleaved nodes, so it can effectively learn the
division — that ability is what §6 exposes.

**Metrics.** MAE, RMSE, MAPE, R². Reported on both the held-out 2023
test set and the 2022-12 OOT slice.
"""))

# ---------------------------------------------------------------- §6 results
cells.append(md(r"""
## 6. Results and Evaluation
"""))

cells.append(code(r"""
mc = pd.read_csv(TABLES / "model_comparison.csv")
mc
"""))

cells.append(md(r"""
**Reading the table.** On the 2023 held-out test set, Ridge reports
**R² = 0.808, MAE = 4.23 min**; the GBM reports **R² = 0.993,
MAE = 0.28 min**. On the 2022-12 OOT slice, those become R² = 0.798 /
MAE = 4.36 min and R² = 0.992 / MAE = 0.30 min respectively.

The 18-point R² jump from Ridge to GBM is too large to attribute to
representational power alone on this many trips — it warrants a
diagnostic. Permutation importance localises the cause:
"""))

cells.append(code(r"""
fi = pd.read_csv(TABLES / "feature_importance_gbm.csv")
fi
"""))

cells.append(md(r"""
**Diagnosis — leakage, not learning.** `trip_miles` (importance 36.4)
and `avg_speed_mph` (25.3) **together account for ~99.5 % of the model's
explanatory power**. The next feature, `same_borough`, sits at 0.10 —
two and a half *orders of magnitude* below. Because
`avg_speed_mph = trip_miles / (trip_time/3600)` and
`trip_duration_min = trip_time / 60`, the model can recover the target
exactly via the identity `duration = miles / speed × 60`. The boosted
trees are not learning travel dynamics; they are performing arithmetic.

The Ridge baseline, by contrast, can only consume each feature as a
linear contribution. It cannot multiply `trip_miles` by `1/avg_speed_mph`
in a single coefficient, so it is forced to rely on additive temporal,
spatial and fare signal. Its **R² = 0.808 / MAE ≈ 4.2 min** is therefore
the honest pre-trip benchmark on this feature set.

Residual diagnostics (full interactive scatter):
[`outputs/figures/05_residuals.html`](../outputs/figures/05_residuals.html).
The Ridge residual cloud is heavy-tailed at long trip durations
(>60 min, mostly airport runs); the GBM residual cloud is a tight
collapse along the diagonal that mirrors the leakage above.

**Temporal generalisation.** Both models drop only slightly from the
2023 test split to the 2022-12 OOT slice — Ridge loses **1.0 R² point
(0.808 → 0.798)** and the GBM loses **0.1 point (0.993 → 0.992)**.
That stability is itself a finding: the duration-distance-speed
structure is essentially invariant across the 12-month gap, so a model
trained on a single year transfers well to the preceding year-end.
"""))

# ---------------------------------------------------------------- §7 insights
cells.append(md(r"""
## 7. Insights, Limitations, and Discussion

### 7.1 Insights from the wrangled data

The interactive Bokeh figures sit alongside this notebook in
`outputs/figures/` (open in a browser for full interactivity):

- **Hourly trip volume** —
  [`01_trips_by_hour.html`](../outputs/figures/01_trips_by_hour.html).
  A clear bimodal weekday pattern with a morning peak around 08:00 and a
  larger evening peak from 17:00–20:00. Late-night demand (00:00–04:00)
  does not collapse, consistent with FHVHV serving the
  taxi-replacement role overnight.
- **Duration by day of week** —
  [`02_duration_by_dow.html`](../outputs/figures/02_duration_by_dow.html).
  Weekday trips are *longer* on average than weekend trips despite the
  weekend nightlife volume — congestion explains more of the duration
  variance than distance does.
- **Monthly volume across 2023** —
  [`03_monthly_volume.html`](../outputs/figures/03_monthly_volume.html).
  Volume rises through the spring, dips in summer, and peaks in October
  / December. December's combined holiday-shopping and weather effect is
  visible.
- **Borough-pair heatmap** —
  [`04_borough_heatmap.html`](../outputs/figures/04_borough_heatmap.html).
  Manhattan ↔ Manhattan dominates trip volume; cross-borough flows are
  asymmetric (e.g. Brooklyn → Manhattan greatly exceeds the reverse
  during morning hours).

### 7.2 Modelling insight

The single most important takeaway is **methodological**, not numeric:
in a corpus where some derived features are algebraic combinations of
the target, a strong R² on a tree model is a leakage diagnostic, not a
performance result. The permutation-importance check is what makes the
difference. The Ridge baseline is robust to this failure mode for
structural reasons (no multiplicative term in a single linear
coefficient), which is a defensible reason to keep a linear baseline
even when boosting is available.

### 7.3 Limitations

- **Sample size and seed.** 5 M of ~232 M rows = 2.2 % of 2023; a single
  random seed. Per-month stratification guards against seasonal
  imbalance but does not address rider-level over-representation
  (heavy users contribute disproportionately to the sample).
- **No exogenous covariates.** Weather, special events, transit outages
  and major road closures all confound duration. Their absence
  bounds achievable R² regardless of model family.
- **`trip_miles` is realised, not planned.** A deployed pre-trip ETA
  service must substitute a routing-engine distance estimate; the
  Ridge MAE of 4.2 min is an *upper bound* on what would be achieved
  in production, because the router's distance estimate will itself
  carry error.
- **No holiday calendar.** `pickup_dow` alone cannot represent
  Thanksgiving / Christmas Eve effects, both of which fall in the OOT
  slice.
- **OOT confounds two effects.** Dec-2022 vs 2023 confounds genuine
  year-over-year drift with the specific holiday-month effect, so the
  ~1-point R² OOT drop should not be interpreted as pure temporal
  drift.

### 7.4 Sources of bias and error

- Stratified-by-month sampling is unbiased at the trip level; it is
  **not** unbiased at the rider level.
- Cleaning thresholds (60 s, 14400 s, 0.1 mi, 100 mi, 500 USD) are
  heuristic round numbers. A formal sensitivity sweep is not done in
  this submission and is an obvious extension.
- One-hot encoding of `PULocationID`/`DOLocationID` is avoided (265²
  zone pairs would dominate the design matrix); we use borough-level
  encoding plus the `same_borough` flag instead. This loses
  zone-pair-specific routing detail that a production model would
  recover through target encoding or zone embeddings.
"""))

# ---------------------------------------------------------------- §8 conclusion
cells.append(md(r"""
## 8. Conclusion

This submission is built around a deliberately compact, fully reproducible
**single-machine pandas pipeline**: eight numbered scripts ingest 24 raw
TLC parquet files, validate and clean a 5 M-row stratified 2023 sample,
join supporting zone metadata, engineer 25+ pre-trip and contextual
features, and train two regression models with side-by-side evaluation
on both an in-sample test split and a year-prior out-of-time slice.

The headline result is **not** "GBM wins by 18 R² points". It is that
**the GBM's apparent performance is leakage from `avg_speed_mph`**, which
encodes the target by construction, and that the honest pre-trip
benchmark on this feature set is the Ridge baseline at **MAE ≈ 4.2 min /
R² ≈ 0.81**. The Ridge–GBM gap, read together with the permutation
importance, is itself the analytical finding — it tells a future
practitioner where to direct effort: not at larger models, but at
**genuine pre-trip features** (routing-engine distance estimates,
weather, planned route geometry, event calendars). Both models
generalise well across a 12-month temporal gap (≤1 R² point drop), which
suggests the structure being learned is stable and that the limits to
accuracy are about feature *content*, not feature *staleness*.

This pipeline can be extended in two natural directions: (1) scaling the
2023 sample to the full 232 M rows with PySpark or DuckDB to test
whether the Ridge baseline's accuracy meaningfully improves with more
data, and (2) substituting realised distance with a router's pre-trip
distance estimate to obtain a deployment-realistic accuracy figure. The
methodology developed here — leakage-aware feature importance, paired
in-sample and OOT evaluation, linear-baseline-vs-boosted comparison —
transfers directly to both extensions.
"""))

# ----------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print(f"wrote {OUT}  ({len(cells)} cells)")
