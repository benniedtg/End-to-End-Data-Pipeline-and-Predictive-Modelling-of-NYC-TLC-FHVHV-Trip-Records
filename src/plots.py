"""Bokeh figures. Each function returns a `figure` instance. Saving is the script's job."""
from __future__ import annotations
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool


def trips_by_hour(df: pd.DataFrame):
    counts = df.groupby("pickup_hour").size().rename("n").reset_index()
    p = figure(title="FHVHV trips by pickup hour (2023 sample)",
               x_axis_label="Pickup hour", y_axis_label="Trips",
               width=720, height=320)
    p.vbar(x="pickup_hour", top="n", source=ColumnDataSource(counts), width=0.8)
    return p


def duration_by_dow(df: pd.DataFrame):
    g = df.groupby("pickup_dow")["trip_duration_min"].median().reset_index()
    p = figure(title="Median trip duration by day-of-week",
               x_axis_label="Day of week (0=Mon)", y_axis_label="Median minutes",
               width=720, height=320)
    p.line(x="pickup_dow", y="trip_duration_min", source=ColumnDataSource(g), line_width=2)
    p.scatter(x="pickup_dow", y="trip_duration_min", source=ColumnDataSource(g), size=8)
    return p


def monthly_volume(df: pd.DataFrame):
    g = df.groupby("pickup_month").size().rename("n").reset_index()
    p = figure(title="Monthly trip volume (2023 sample)",
               x_axis_label="Month", y_axis_label="Trips", width=720, height=320)
    p.vbar(x="pickup_month", top="n", source=ColumnDataSource(g), width=0.8)
    return p


def borough_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="pu_borough", columns="do_borough",
                           values="trip_duration_min", aggfunc="median")
    melt = pivot.reset_index().melt(id_vars="pu_borough",
                                    var_name="do_borough", value_name="median_min")
    p = figure(
        title="Median duration by PU x DO borough",
        x_range=list(pivot.columns.astype(str)), y_range=list(pivot.index.astype(str)),
        x_axis_label="Drop-off borough", y_axis_label="Pickup borough",
        width=720, height=420, tools="hover",
    )
    p.rect(x="do_borough", y="pu_borough", width=1, height=1,
           source=ColumnDataSource(melt), fill_color="#5b9aa0", line_color="white")
    hover: HoverTool = p.select_one(HoverTool)
    hover.tooltips = [("pu", "@pu_borough"), ("do", "@do_borough"), ("median min", "@median_min{0.0}")]
    return p


def residual_scatter(y_true, y_pred, sample_n: int = 30_000, seed: int = 42):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)
    df["residual"] = df["y_pred"] - df["y_true"]
    p = figure(title="Residuals vs predicted (GBM, test set)",
               x_axis_label="Predicted (min)", y_axis_label="Residual (min)",
               width=720, height=360)
    p.scatter(x="y_pred", y="residual", source=ColumnDataSource(df), size=2, alpha=0.2)
    return p
