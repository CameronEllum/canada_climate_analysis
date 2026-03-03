"""Climate report generation with Plotly visualizations."""

from __future__ import annotations

import datetime
from pathlib import Path

import jinja2
import polars as pl
from constants import MONTH_LABELS
from constants import SEASON_LABELS
from constants import YEAR_LABELS
from report_plots import calculate_trendline
from report_plots import create_precipitation_plot
from report_plots import create_station_map
from report_plots import create_temperature_plot


def aggregate_data(
    daily_df: pl.DataFrame,
    stations_df: pl.DataFrame,
    max_temp: bool = False,
    min_temp: bool = False,
    period: str = "monthly",
    percentiles: list[int] | None = None,
) -> pl.DataFrame:
    """Aggregate daily data to monthly statistics.

    Determines the source column based on max_temp or min_temp flags.
    """
    if max_temp:
        target_col = "temp_max"
        min_col = "temp_max"
        max_col = "temp_max"
    elif min_temp:
        target_col = "temp_min"
        min_col = "temp_min"
        max_col = "temp_min"
    else:
        target_col = "temp_mean"
        min_col = "temp_min"
        max_col = "temp_max"

    # Join with stations to get requested_location
    df = daily_df.join(
        stations_df.select(["id", "requested_location"]),
        left_on="station_id",
        right_on="id",
    ).with_columns(
        period_idx=pl.when(period == "monthly")
        .then(pl.col("month"))
        .when(period == "seasonally")
        .then(
            pl.when(pl.col("month").is_in([12, 1, 2]))
            .then(1)
            .when(pl.col("month").is_in([3, 4, 5]))
            .then(2)
            .when(pl.col("month").is_in([6, 7, 8]))
            .then(3)
            .otherwise(4)
        )
        .otherwise(pl.lit(1))
    )

    # Ensure hover-required percentiles are included
    if percentiles is None:
        percentiles = [50]

    required = {0, 25, 50, 75, 100}
    p_set = sorted(
        list(
            set([(100 - p) for p in percentiles])  # Percentiles above the mean
            | set(percentiles)  # Percentiles below the mean
            | required  # Required percentiles
        )
    )

    agg_exprs = [
        pl.col(target_col).mean().alias("temp_mean"),
        pl.col(target_col).median().alias("temp_median"),
        pl.col(min_col).min().alias("temp_min_abs"),
        pl.col(max_col).max().alias("temp_max_abs"),
        (pl.col("precip_total").sum() / pl.col("station_id").n_unique()).alias(
            "precip_total"
        ),
        pl.col("precip_total").median().alias("precip_median"),
    ]

    for p in p_set:
        q = p / 100
        agg_exprs.append(pl.col(target_col).quantile(q).alias(f"temp_p{p}"))
        agg_exprs.append(
            pl.col("precip_total").quantile(q).alias(f"precip_p{p}")
        )

    return df.group_by(["requested_location", "year", "period_idx"]).agg(
        agg_exprs
    )


def render_template(
    locations: list[str],
    radius: float,
    plots: list[str],
    period_labels: list[str],
    traces_per_month_temp: int,
    traces_per_month_precip: int,
    num_ribbons: int,
    num_locations: int,
    show_trend: bool,
    show_median: bool,
) -> str:
    """Render the HTML report using Jinja2."""
    template_path = Path(__file__).parent / "template.html"
    with open(template_path, "r", encoding="utf-8") as f:
        template = jinja2.Template(f.read())

    f_url = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"

    return template.render(
        location=" & ".join([loc.split(",")[0] for loc in locations]),
        radius=radius,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        plots=plots,
        f_url=f_url,
        months=period_labels,
        traces_per_month_temp=traces_per_month_temp,
        traces_per_month_precip=traces_per_month_precip,
        num_ribbons=num_ribbons,
        num_locations=num_locations,
        show_trend=show_trend,
        show_median=show_median,
    )


def calculate_anomalies(stats_df: pl.DataFrame) -> pl.DataFrame:
    """Add anomaly and trend columns to period statistics."""
    avg_series = stats_df["avg"]
    lt_mean = avg_series.mean()
    if lt_mean is None:
        lt_mean = 0.0

    x_vals = [float(xi) for xi in stats_df["year"].to_list()]
    y_vals = avg_series.to_list()
    # Filter out None values for trend calculation
    valid_y = [y for y in y_vals if y is not None]

    trend_y = None
    if len(valid_y) >= 2:
        trend_y = calculate_trendline(x_vals, y_vals)

    if trend_y:
        series_trend = pl.Series(name="trend", values=trend_y)
        return stats_df.with_columns(
            trend=series_trend,
            anomaly=pl.col("avg") - series_trend,
        )
    else:
        return stats_df.with_columns(
            trend=pl.lit(lt_mean),
            anomaly=pl.col("avg") - lt_mean,
        )


def _calculate_period_stats(
    merged_df: pl.DataFrame,
    period_idx: int,
    metric: str,
    location: str | None = None,
) -> pl.DataFrame | None:
    """Calculate statistics for a specific period (month/season/year)."""

    p_df = merged_df.filter(pl.col("period_idx") == period_idx)

    if location:
        p_df = p_df.filter(pl.col("requested_location") == location)

    p_df = p_df.sort("year")

    if p_df.is_empty():
        return None

    if metric == "temperature":
        avg_col = "temp_mean"
        median_col = "temp_median"
        min_col = "temp_min_abs"
        max_col = "temp_max_abs"
        prefix = "temp_p"
    else:  # precipitation
        avg_col = "precip_total"
        median_col = "precip_median"
        min_col = "precip_total"
        max_col = "precip_total"
        prefix = "precip_p"

    p_cols = [c for c in p_df.columns if c.startswith(prefix)]

    agg_exprs = [
        pl.col(avg_col).mean().alias("avg"),
        pl.col(median_col).mean().alias("median"),
        pl.col(min_col).min().alias("min"),
        pl.col(max_col).max().alias("max"),
    ]

    for c in p_cols:
        agg_exprs.append(pl.col(c).mean().alias(c.split("_")[-1]))

    stats_df = p_df.group_by("year").agg(agg_exprs).sort("year")

    return calculate_anomalies(stats_df)


def generate_report(
    daily_df: pl.DataFrame,
    stations_df: pl.DataFrame,
    locations: list[str],
    radius: float,
    show_trend: bool = False,
    show_median: bool = False,
    show_anomaly: bool = True,
    max_temp: bool = False,
    min_temp: bool = False,
    period: str = "monthly",
    ribbon_percentiles: list[int] | None = None,
) -> str:
    """Aggregate daily data to period and generate HTML report."""

    if ribbon_percentiles is None:
        ribbon_percentiles = list(range(0, 101, 5))

    merged_df = aggregate_data(
        daily_df,
        stations_df,
        max_temp,
        min_temp,
        period,
        percentiles=ribbon_percentiles,
    )

    if period == "monthly":
        period_labels = MONTH_LABELS
    elif period == "seasonally":
        period_labels = SEASON_LABELS
    else:  # yearly
        period_labels = YEAR_LABELS

    temp_stats_map = {}
    precip_stats_map = {}

    for p_idx in range(1, len(period_labels) + 1):
        for loc in locations:
            t_stats = _calculate_period_stats(
                merged_df, p_idx, "temperature", location=loc
            )
            if t_stats is not None:
                temp_stats_map[(p_idx, loc)] = t_stats

            p_stats = _calculate_period_stats(
                merged_df, p_idx, "precipitation", location=loc
            )
            if p_stats is not None:
                precip_stats_map[(p_idx, loc)] = p_stats

    # Create plots with pre-calculated data
    fig_temp = create_temperature_plot(
        temp_stats_map,
        period_labels,
        show_trend,
        show_median,
        show_anomaly,
        max_temp,
        min_temp,
        locations=locations,
        period_type=period,
        percentiles=ribbon_percentiles,
    )

    fig_precip = create_precipitation_plot(
        precip_stats_map,
        period_labels,
        show_trend,
        show_anomaly,
        locations=locations,
        period_type=period,
    )

    fig_map = create_station_map(stations_df, daily_df)

    html_sections = [
        fig_temp.to_html(
            include_plotlyjs="cdn",
            div_id="chart-temp",
            full_html=False,
            config={"modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        ),
        fig_precip.to_html(
            include_plotlyjs=False,
            div_id="chart-precip",
            full_html=False,
            config={"modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        ),
        fig_map.to_html(
            include_plotlyjs=False,
            div_id="chart-map",
            full_html=False,
        ),
    ]

    ribbon_pairs = [p for p in ribbon_percentiles if p < 50]
    traces_per_loc_temp = 2 * len(ribbon_pairs) + 3  # obs, trend, median
    traces_per_month_temp = traces_per_loc_temp * len(locations)
    traces_per_loc_precip = 2  # obs, trend
    traces_per_month_precip = traces_per_loc_precip * len(locations)

    return render_template(
        locations,
        radius,
        html_sections,
        period_labels,
        traces_per_month_temp,
        traces_per_month_precip,
        len(ribbon_pairs),
        len(locations),
        show_trend,
        show_median,
    )
