"""Climate report generation with Plotly visualizations."""

from __future__ import annotations

import datetime
from pathlib import Path

import jinja2
import plotly.graph_objects as go
import polars as pl
from scipy import stats


def calculate_trendline(x: list[float], y: list[float]) -> list[float] | None:
    """Calculate simple linear trendline using scipy.

    Filters out None values from the input data before calculating the trend.
    Returns None if insufficient valid data points remain.
    """
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None

    # Filter out None values
    valid_pairs = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]

    if len(valid_pairs) < 2:
        return None

    # Unzip the valid pairs
    x_valid, y_valid = zip(*valid_pairs)

    # Use scipy's linregress for robust linear regression
    result = stats.linregress(x_valid, y_valid)
    slope = result.slope
    intercept = result.intercept

    # Return trend values for ALL original x values (including those with None y)
    return [slope * xi + intercept for xi in x]


def create_modern_theme(fig: go.Figure) -> None:
    """Apply a clean, modern theme."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter, sans-serif",
        font_size=12,
        xaxis=dict(
            showgrid=True,
            gridcolor="#f0f0f0",
            linecolor="#333",
            linewidth=1,
            ticks="outside",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f0f0f0",
            linecolor="#333",
            linewidth=1,
            ticks="outside",
        ),
    )


def aggregate_to_monthly(daily_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate daily data to monthly statistics."""
    return daily_df.group_by(["station_id", "year", "month"]).agg(
        [
            pl.col("temp_mean").mean().alias("temp_mean"),
            pl.col("temp_min").min().alias("temp_min_abs"),
            pl.col("temp_max").max().alias("temp_max_abs"),
            pl.col("precip").sum().alias("precip_total"),
            pl.col("temp_mean").quantile(0.25).alias("temp_q1"),
            pl.col("temp_mean").quantile(0.75).alias("temp_q3"),
            pl.col("precip").quantile(0.25).alias("precip_q1"),
            pl.col("precip").quantile(0.75).alias("precip_q3"),
        ]
    )


def render_template(
    location_name: str,
    radius: float,
    html_sections: list[str],
    months: list[str],
    traces_per_month: int,
    show_trend: bool,
    shade_deviation: bool,
) -> str:
    """Render the HTML report from template."""
    template_path = Path(__file__).parent / "template.html"
    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    f_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Inter:wght@400;600&display=swap"
    )

    return jinja2.Template(template_content).render(
        location=location_name,
        radius=radius,
        date=datetime.date.today().isoformat(),
        plots=html_sections,
        f_url=f_url,
        months=months,
        traces_per_month=traces_per_month,
        show_trend=show_trend,
        show_dev=shade_deviation,
    )


def generate_report(
    daily_df: pl.DataFrame,
    stations_df: pl.DataFrame,
    location_name: str,
    radius: float,
    show_trend: bool = False,
    shade_deviation: bool = False,
    show_anomaly: bool = True,
) -> str:
    """Aggregate daily data to monthly and generate HTML report."""
    # Import here to avoid circular dependency
    from report_plots import create_precipitation_plot
    from report_plots import create_station_map
    from report_plots import create_temperature_plot

    monthly_df = aggregate_to_monthly(daily_df)

    months = [
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

    # Create temperature plot
    fig_temp = create_temperature_plot(
        monthly_df, months, show_trend, shade_deviation, show_anomaly
    )

    # Create precipitation plot
    fig_precip = create_precipitation_plot(
        monthly_df, months, show_trend, shade_deviation, show_anomaly
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

    traces_per_month = 4 if shade_deviation else 3

    return render_template(
        location_name,
        radius,
        html_sections,
        months,
        traces_per_month,
        show_trend,
        shade_deviation,
    )
