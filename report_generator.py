"""Climate report generation with Plotly visualizations."""

from __future__ import annotations

import datetime
from pathlib import Path

import jinja2
import polars as pl


def aggregate_data(
    daily_df: pl.DataFrame,
    stations_df: pl.DataFrame,
    max_temp: bool = False,
    min_temp: bool = False,
    period: str = "monthly",
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

    return df.group_by(
        ["requested_location", "station_id", "year", "period_idx"]
    ).agg(
        [
            pl.col(target_col).mean().alias("temp_mean"),
            pl.col(min_col).min().alias("temp_min_abs"),
            pl.col(max_col).max().alias("temp_max_abs"),
            pl.col("precip").sum().alias("precip_total"),
            pl.col(target_col).quantile(0.25).alias("temp_q1"),
            pl.col(target_col).quantile(0.75).alias("temp_q3"),
            pl.col("precip").quantile(0.25).alias("precip_q1"),
            pl.col("precip").quantile(0.75).alias("precip_q3"),
        ]
    )


def render_template(
    locations: list[str],
    radius: float,
    plots: list[str],
    period_labels: list[str],
    traces_per_month: int,
    show_trend: bool,
    shade_deviation: bool,
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
        traces_per_month=traces_per_month,
        show_trend=show_trend,
        show_dev=shade_deviation,
    )


def generate_report(
    daily_df: pl.DataFrame,
    stations_df: pl.DataFrame,
    locations: list[str],
    radius: float,
    show_trend: bool = False,
    std_dev: bool = False,
    show_anomaly: bool = True,
    max_temp: bool = False,
    min_temp: bool = False,
    period: str = "monthly",
) -> str:
    """Aggregate daily data to period and generate HTML report."""
    # Import here to avoid circular dependency
    from report_plots import create_precipitation_plot
    from report_plots import create_station_map
    from report_plots import create_temperature_plot

    merged_df = aggregate_data(
        daily_df, stations_df, max_temp, min_temp, period
    )

    if period == "monthly":
        period_labels = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    elif period == "seasonally":
        period_labels = [
            "Winter (DJF)",
            "Spring (MAM)",
            "Summer (JJA)",
            "Fall (SON)",
        ]
    else:  # yearly
        period_labels = ["Annual"]

    # Create temperature plot
    fig_temp = create_temperature_plot(
        merged_df,
        period_labels,
        show_trend,
        std_dev,
        show_anomaly,
        max_temp,
        min_temp,
        locations=locations,
        period_type=period,
    )

    # Create precipitation plot
    fig_precip = create_precipitation_plot(
        merged_df,
        period_labels,
        show_trend,
        std_dev,
        show_anomaly,
        locations=locations,
        period_type=period,
    )

    # Create station map
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

    # Each location adds 4 traces (shading, range, trend, main)
    traces_per_month = 4 * len(locations)

    return render_template(
        locations,
        radius,
        html_sections,
        period_labels,
        traces_per_month,
        show_trend,
        std_dev,
    )
