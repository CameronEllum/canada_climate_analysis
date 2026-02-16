"""Plot creation functions for climate reports."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from report_generator import calculate_trendline
from report_generator import create_modern_theme


def _calculate_monthly_stats(
    monthly_df: pl.DataFrame, month_idx: int, metric: str
) -> pl.DataFrame | None:
    """Calculate statistics for a specific month."""
    m_df = monthly_df.filter(pl.col("month") == month_idx).sort("year")

    if m_df.is_empty():
        return None

    if metric == "temperature":
        return (
            m_df.group_by("year")
            .agg(
                [
                    pl.col("temp_mean").mean().alias("avg"),
                    pl.col("temp_mean").median().alias("median"),
                    pl.col("temp_q1").mean().alias("q1"),
                    pl.col("temp_q3").mean().alias("q3"),
                    pl.col("temp_min_abs").min().alias("min"),
                    pl.col("temp_max_abs").max().alias("max"),
                ]
            )
            .sort("year")
        )
    else:  # precipitation
        return (
            m_df.group_by("year")
            .agg(
                [
                    pl.col("precip_total").mean().alias("avg"),
                    pl.col("precip_total").median().alias("median"),
                    pl.col("precip_q1").mean().alias("q1"),
                    pl.col("precip_q3").mean().alias("q3"),
                    pl.col("precip_total").min().alias("min"),
                    pl.col("precip_total").max().alias("max"),
                ]
            )
            .sort("year")
        )


def _add_anomaly_columns(stats: pl.DataFrame) -> pl.DataFrame:
    """Add anomaly column to monthly statistics.

    Trend Calculation Methodology:
    ==============================
    1. Input: stats DataFrame contains yearly averages for a specific month
       - Each row represents one year's data for that month
       - 'avg' column = mean of all station observations for that month/year

    2. Linear Trend Calculation:
       - Fits a linear regression line through the yearly averages
       - Formula: trend(year) = slope * year + intercept
       - This represents the long-term climate trend for this specific month

    3. Anomaly Calculation:
       - Anomaly = actual yearly average - trend value for that year
       - Positive anomaly = warmer/wetter than trend
       - Negative anomaly = cooler/drier than trend

    Example for January:
    - 1950 January avg = -5.2°C, trend = -5.0°C → anomaly = -0.2°C
    - 2020 January avg = -3.1°C, trend = -4.0°C → anomaly = +0.9°C
    """
    lt_mean = stats["avg"].mean()
    x_vals = [float(xi) for xi in stats["year"].to_list()]
    y_vals = stats["avg"].to_list()
    trend_y = calculate_trendline(x_vals, y_vals)

    if trend_y:
        # Use linear trend as baseline for anomaly calculation
        series_trend = pl.Series(name="trend", values=trend_y)
        return stats.with_columns(
            trend=series_trend,
            anomaly=pl.col("avg") - series_trend,
        )
    else:
        # Fallback to long-term mean if trend calculation fails
        return stats.with_columns(
            trend=pl.lit(lt_mean),
            anomaly=pl.col("avg") - lt_mean,
        )


def _create_shading_trace(
    x: list,
    y: list,
    month_idx: int,
    std_dev: bool,
    fillcolor: str,
) -> go.Scatter:
    """Create standard deviation shading trace."""
    y_upper, y_lower = [], []
    if x and y:
        x_vals = [float(xi) for xi in x]
        trend_y = calculate_trendline(x_vals, y)
        if trend_y:
            # Filter out None values when calculating residuals
            resids = [yi - tyi for yi, tyi in zip(y, trend_y) if yi is not None]
            if resids:  # Only calculate std if we have valid residuals
                std_resid = (sum(r**2 for r in resids) / len(resids)) ** 0.5
                y_upper = [ty + std_resid for ty in trend_y]
                y_lower = [ty - std_resid for ty in trend_y]

    return go.Scatter(
        x=x + x[::-1],
        y=y_upper + y_lower[::-1],
        fill="toself",
        fillcolor=fillcolor,
        line=dict(color="rgba(0,0,0,0)"),
        name="Std Dev (Trend-rel.)",
        visible=(month_idx == 1 and std_dev),
        showlegend=std_dev,
        hoverinfo="skip",
    )


def _create_trend_trace(
    x: list, y: list, month_idx: int, show_trend: bool
) -> go.Scatter:
    """Create trendline trace."""
    x_vals = [float(xi) for xi in x]
    trend_y = calculate_trendline(x_vals, y) if x and y else None

    return go.Scatter(
        x=x,
        y=trend_y if trend_y else [],
        mode="lines",
        name="Linear Trend",
        visible=(month_idx == 1 and show_trend),
        line=dict(width=1, color="red"),
        showlegend=show_trend,
        hoverinfo="skip",
    )


def create_temperature_plot(
    monthly_df: pl.DataFrame,
    months: list[str],
    show_trend: bool,
    std_dev: bool,
    show_anomaly: bool,
    max_temp: bool = False,
    min_temp: bool = False,
) -> go.Figure:
    """Create temperature analysis plot."""
    fig = go.Figure()
    if max_temp:
        mean_label = "Mean Max"
        title_text = "Monthly Maximum Temperature Analysis"
    elif min_temp:
        mean_label = "Mean Min"
        title_text = "Monthly Minimum Temperature Analysis"
    else:
        mean_label = "Mean"
        title_text = "Monthly Temperature Analysis"

    for m_idx in range(1, 13):
        stats = _calculate_monthly_stats(monthly_df, m_idx, "temperature")

        if stats is not None:
            stats = _add_anomaly_columns(stats)

        x = stats["year"].to_list() if stats is not None else []
        y = stats["avg"].to_list() if stats is not None else []
        anom_list = stats["anomaly"].to_list() if stats is not None else []
        c_data = (
            stats[
                [
                    "q1",
                    "q3",
                    "min",
                    "max",
                    "anomaly",
                    "median",
                    "trend",
                ]
            ].rows()
            if stats is not None
            else []
        )

        # Trace 0: Shading
        fig.add_trace(
            _create_shading_trace(
                x, y, m_idx, std_dev, "rgba(200, 200, 200, 0.15)"
            )
        )

        # Trace 1: Q1/Q3 Range
        q3 = stats["q3"] if stats is not None else []
        q1 = stats["q1"] if stats is not None else []
        avg = stats["avg"] if stats is not None else []
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Spread (Q1-Q3)",
                visible=(m_idx == 1),
                marker=dict(size=0),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(q3 - avg).to_list() if stats is not None else [],
                    arrayminus=(avg - q1).to_list()
                    if stats is not None
                    else [],
                    width=0,
                    thickness=2,
                    color="rgba(0, 0, 0, 0.35)",
                ),
                showlegend=True,
                hoverinfo="skip",
            )
        )

        # Trace 2: Trendline
        fig.add_trace(_create_trend_trace(x, y, m_idx, show_trend))

        # Trace 3: Main Data
        # Replace None values with 0 for color mapping
        m_color = (
            [a if a is not None else 0 for a in anom_list]
            if show_anomaly
            else "#2c3e50"
        )
        m_cscale = "RdBu_r" if show_anomaly else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=c_data,
                mode="lines+markers",
                name="Observations",
                visible=(m_idx == 1),
                marker=dict(
                    size=8,
                    color=m_color,
                    colorscale=m_cscale,
                    cmid=0,
                    line=dict(width=1, color="white"),
                    colorbar=dict(
                        title=dict(text="Anomaly (°C)", side="top"),
                        orientation="h",
                        x=0.5,
                        y=-0.18,
                        yanchor="top",
                        xanchor="center",
                        thickness=15,
                        len=0.5,
                    )
                    if show_anomaly
                    else None,
                ),
                line=dict(width=1, color="rgba(0,0,0,0.2)"),
                showlegend=True,
                hovertemplate=(
                    "<b>Year: %{x}</b><br>" + mean_label + ": %{y:.1f}°C<br>"
                    "Median: %{customdata[5]:.1f}°C<br>"
                    "Trend Mean: %{customdata[6]:.1f}°C<br>"
                    "Mean Anomaly: %{customdata[4]:.1f}°C<br>"
                    "Minimum: %{customdata[2]:.1f}°C<br>"
                    "Maximum: %{customdata[3]:.1f}°C<br>"
                    "25th Percentile: %{customdata[0]:.1f}°C<br>"
                    "75th Percentile: %{customdata[1]:.1f}°C<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            y=0.96,
            xanchor="center",
        ),
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02
        ),
        margin=dict(l=60, r=160, t=80, b=100),
        dragmode="pan",
        height=600,
    )
    create_modern_theme(fig)
    return fig


def create_precipitation_plot(
    monthly_df: pl.DataFrame,
    months: list[str],
    show_trend: bool,
    std_dev: bool,
    show_anomaly: bool,
) -> go.Figure:
    """Create precipitation analysis plot."""
    fig = go.Figure()

    for m_idx in range(1, 13):
        stats = _calculate_monthly_stats(monthly_df, m_idx, "precipitation")

        if stats is not None:
            stats = _add_anomaly_columns(stats)

        x = stats["year"].to_list() if stats is not None else []
        y = stats["avg"].to_list() if stats is not None else []
        anom_list = stats["anomaly"].to_list() if stats is not None else []
        c_data = (
            stats[
                [
                    "q1",
                    "q3",
                    "min",
                    "max",
                    "anomaly",
                    "median",
                    "trend",
                ]
            ].rows()
            if stats is not None
            else []
        )

        # Trace 0: Shading
        fig.add_trace(
            _create_shading_trace(
                x, y, m_idx, std_dev, "rgba(100, 150, 200, 0.1)"
            )
        )

        # Trace 1: Range (Removed for Precipitation)
        fig.add_trace(go.Scatter(x=[], y=[], visible=False, showlegend=False))

        # Trace 2: Trend
        fig.add_trace(_create_trend_trace(x, y, m_idx, show_trend))

        # Trace 3: Main
        # Replace None values with 0 for color mapping
        m_color = (
            [a if a is not None else 0 for a in anom_list]
            if show_anomaly
            else "#1a5fb4"
        )
        m_cscale = "BrBG" if show_anomaly else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=c_data,
                mode="lines+markers",
                name="Observations",
                visible=(m_idx == 1),
                marker=dict(
                    size=8,
                    color=m_color,
                    colorscale=m_cscale,
                    cmid=0,
                    line=dict(width=1, color="white"),
                    colorbar=dict(
                        title=dict(text="Anomaly (mm)", side="top"),
                        orientation="h",
                        x=0.5,
                        y=-0.18,
                        yanchor="top",
                        xanchor="center",
                        thickness=15,
                        len=0.5,
                    )
                    if show_anomaly
                    else None,
                ),
                line=dict(width=1, color="rgba(0,0,0,0.2)"),
                showlegend=True,
                hovertemplate=(
                    "<b>Year: %{x}</b><br>Total: %{y:.1f} mm<br>"
                    "Median: %{customdata[5]:.1f} mm<br>"
                    "Trend Mean: %{customdata[6]:.1f} mm<br>"
                    "Mean Anomaly: %{customdata[4]:.1f} mm<br>"
                    "Minimum: %{customdata[2]:.1f} mm<br>"
                    "Maximum: %{customdata[3]:.1f} mm<br>"
                    "25th Percentile: %{customdata[0]:.1f} mm<br>"
                    "75th Percentile: %{customdata[1]:.1f} mm<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Monthly Precipitation Analysis",
            x=0.5,
            y=0.96,
            xanchor="center",
        ),
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02
        ),
        margin=dict(l=60, r=160, t=80, b=100),
        dragmode="pan",
        height=600,
    )
    create_modern_theme(fig)
    return fig


def create_station_map(
    stations_df: pl.DataFrame,
    daily_df: pl.DataFrame,
) -> go.Figure:
    """Create a map showing historical and active climate stations."""
    fig = go.Figure()

    # Calculate year ranges for each station
    max_year = daily_df["year"].max()
    range_df = daily_df.group_by("station_id").agg(
        [
            pl.col("year").min().alias("min_y"),
            pl.col("year").max().alias("max_y"),
        ]
    )

    # Join station info with year ranges
    map_stations = stations_df.join(
        range_df, left_on="id", right_on="station_id"
    )
    map_stations = map_stations.with_columns(
        is_current=pl.col("max_y") == max_year
    )

    # Separate into historical and active stations
    curr_df = map_stations.filter(pl.col("is_current"))
    hist_df = map_stations.filter(~pl.col("is_current"))

    for group_df, color, label in [
        (hist_df, "#e74c3c", "Historical"),
        (curr_df, "#2ecc71", "Active"),
    ]:
        if group_df.is_empty():
            continue
        lats, lons, h_text = [], [], []
        for row in group_df.iter_rows(named=True):
            years = f"{int(row['min_y'])} - {int(row['max_y'])}"
            h_text.append(f"<b>{row['name']}</b><br>Dates: {years}")
            lats.append(row["latitude"])
            lons.append(row["longitude"])
        fig.add_trace(
            go.Scattermap(
                lat=lats,
                lon=lons,
                mode="markers",
                name=label,
                marker=go.scattermap.Marker(size=10, color=color, opacity=0.8),
                text=h_text,
                hoverinfo="text",
            )
        )

    if not map_stations.is_empty():
        fig.update_layout(
            map=dict(
                style="carto-positron",
                center=dict(
                    lat=map_stations["latitude"].mean(),
                    lon=map_stations["longitude"].mean(),
                ),
                zoom=8,
            ),
            margin=dict(l=0, r=0, t=50, b=10),
            height=500,
            title=dict(
                text="Climate Stations", x=0.5, y=0.98, xanchor="center"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.7)",
            ),
        )
    return fig
