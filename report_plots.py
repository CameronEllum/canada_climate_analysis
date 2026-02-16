"""Plot creation functions for climate reports."""

from __future__ import annotations

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

    # Return trend values for all x values (including those with None y)
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


def _calculate_period_stats(
    merged_df: pl.DataFrame, period_idx: int, metric: str, location: str = None
) -> pl.DataFrame | None:
    """Calculate statistics for a specific period (month/season/year)."""
    p_df = merged_df.filter(pl.col("period_idx") == period_idx)

    if location:
        p_df = p_df.filter(pl.col("requested_location") == location)

    p_df = p_df.sort("year")

    if p_df.is_empty():
        return None

    prefix = "temp_" if metric == "temperature" else "precip_"
    p_cols = [c for c in p_df.columns if c.startswith(f"{prefix}p")]

    agg_exprs = []
    if metric == "temperature":
        agg_exprs.extend(
            [
                pl.col("temp_mean").mean().alias("avg"),
                pl.col("temp_mean").median().alias("median"),
                pl.col("temp_min_abs").min().alias("min"),
                pl.col("temp_max_abs").max().alias("max"),
            ]
        )
    else:  # precipitation
        agg_exprs.extend(
            [
                pl.col("precip_total").mean().alias("avg"),
                pl.col("precip_total").median().alias("median"),
                pl.col("precip_total").min().alias("min"),
                pl.col("precip_total").max().alias("max"),
            ]
        )

    for c in p_cols:
        p_val = c.split("p")[-1]
        agg_exprs.append(pl.col(c).mean().alias(f"p{p_val}"))

    return p_df.group_by("year").agg(agg_exprs).sort("year")


def _add_anomaly_columns(stats: pl.DataFrame) -> pl.DataFrame:
    """Add anomaly column to period statistics."""
    lt_mean = stats["avg"].mean()
    x_vals = [float(xi) for xi in stats["year"].to_list()]
    y_vals = stats["avg"].to_list()
    trend_y = calculate_trendline(x_vals, y_vals)

    if trend_y:
        series_trend = pl.Series(name="trend", values=trend_y)
        return stats.with_columns(
            trend=series_trend,
            anomaly=pl.col("avg") - series_trend,
        )
    else:
        return stats.with_columns(
            trend=pl.lit(lt_mean),
            anomaly=pl.col("avg") - lt_mean,
        )


# Standard deviation shading removed as per request


def _create_trend_trace(
    x: list, y: list, p_idx: int, show_trend: bool
) -> go.Scatter:
    """Create trendline trace."""
    x_vals = [float(xi) for xi in x]
    trend_y = calculate_trendline(x_vals, y) if x and y else None

    return go.Scatter(
        x=x,
        y=trend_y if trend_y else [],
        mode="lines",
        name="Linear Trend",
        visible=(p_idx == 1 and show_trend),
        line=dict(width=1, color="red"),
        showlegend=show_trend,
        hoverinfo="skip",
    )


def create_temperature_plot(
    merged_df: pl.DataFrame,
    period_labels: list[str],
    show_trend: bool,
    show_anomaly: bool,
    max_temp: bool = False,
    min_temp: bool = False,
    locations: list[str] = None,
    period_type: str = "monthly",
    percentiles: list[int] | None = None,
) -> go.Figure:
    """Create temperature analysis plot."""
    fig = go.Figure()

    if locations is None:
        locations = ["All Stations"]

    colors = [
        "#2c3e50",
        "#e74c3c",
        "#27ae60",
        "#2980b9",
        "#8e44ad",
        "#f39c12",
        "#d35400",
        "#16a085",
    ]

    prefix = (
        "Monthly"
        if period_type == "monthly"
        else ("Seasonal" if period_type == "seasonally" else "Yearly")
    )
    if max_temp:
        mean_label = "Mean Max"
        title_text = f"{prefix} Maximum Temperature Analysis"
    elif min_temp:
        mean_label = "Mean Min"
        title_text = f"{prefix} Minimum Temperature Analysis"
    else:
        mean_label = "Mean"
        title_text = f"{prefix} Temperature Analysis"

    for p_idx in range(1, len(period_labels) + 1):
        for i, loc in enumerate(locations):
            stats_df = _calculate_period_stats(
                merged_df, p_idx, "temperature", location=loc
            )

            if stats_df is not None:
                stats_df = _add_anomaly_columns(stats_df)

            x = stats_df["year"].to_list() if stats_df is not None else []
            y = stats_df["avg"].to_list() if stats_df is not None else []
            anom_list = (
                stats_df["anomaly"].to_list() if stats_df is not None else []
            )
            c_data = (
                stats_df[
                    ["p25", "p75", "min", "max", "anomaly", "median", "trend"]
                ].rows()
                if stats_df is not None
                else []
            )

            color = colors[i % len(colors)]
            loc_prefix = f"{loc.split(',')[0]} - " if len(locations) > 1 else ""

            # Shading color logic (retained for ribbons if needed)
            shading_color = color.replace("#", "")
            r, g, b = (
                int(shading_color[:2], 16),
                int(shading_color[2:4], 16),
                int(shading_color[4:], 16),
            )

            # Ribbons: symmetric pairs
            if percentiles is None:
                percentiles = list(range(0, 101, 5))
            ribbon_pairs = sorted(
                [(p, 100 - p) for p in percentiles if p < 50], reverse=True
            )
            rib_grp = f"ribbons_{i}_{p_idx}"

            for low_p_val, high_p_val in sorted(ribbon_pairs, reverse=False):
                low_p = f"p{low_p_val}"
                high_p = f"p{high_p_val}"
                y_high = (
                    stats_df[high_p].to_list() if stats_df is not None else []
                )
                y_low = (
                    stats_df[low_p].to_list() if stats_df is not None else []
                )

                # Ribbon boundary (Top)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_high,
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        legendgroup=rib_grp,
                        visible=(p_idx == 1),
                        hoverinfo="skip",
                    )
                )

                # Ribbon fill (Bottom)
                # Outer label for legend
                outer_low, outer_high = ribbon_pairs[0]
                label = f"{loc_prefix}Percentiles"
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_low,
                        mode="lines",
                        fill="tonexty",
                        fillcolor=f"rgba({r}, {g}, {b}, 0.05)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name=label,
                        legendgroup=rib_grp,
                        visible=(p_idx == 1),
                        showlegend=(low_p_val == outer_low),
                        hoverinfo="skip",
                    )
                )

            t_trace = _create_trend_trace(x, y, p_idx, show_trend)
            t_trace.name = f"{loc_prefix}{t_trace.name}"
            t_trace.line.color = color
            if len(locations) > 1:
                t_trace.line.dash = "dot"
            fig.add_trace(t_trace)

            if show_anomaly and len(locations) == 1:
                m_color = [a if a is not None else 0 for a in anom_list]
                m_cscale, show_colorbar = "RdBu_r", True
            else:
                m_color, m_cscale, show_colorbar = color, None, False

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    customdata=c_data,
                    mode="lines+markers",
                    name=f"{loc_prefix}Observations",
                    visible=(p_idx == 1),
                    marker=dict(
                        size=8 if len(locations) > 1 else 10,
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
                        if show_colorbar
                        else None,
                    ),
                    line=dict(width=1, color="rgba(0,0,0,0.2)"),
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{loc}</b><br><b>Year: %{{x}}</b><br>"
                        f"{mean_label}: %{{y:.1f}}°C<br>"
                        f"Median: %{{customdata[5]:.1f}}°C<br>"
                        f"Trend Mean: %{{customdata[6]:.1f}}°C<br>"
                        f"Mean Anomaly: %{{customdata[4]:.1f}}°C<br>"
                        f"Minimum: %{{customdata[2]:.1f}}°C<br>"
                        f"Maximum: %{{customdata[3]:.1f}}°C<br>"
                        f"25th Percentile: %{{customdata[0]:.1f}}°C<br>"
                        f"75th Percentile: %{{customdata[1]:.1f}}°C<br><extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, y=0.96, xanchor="center"),
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
    merged_df: pl.DataFrame,
    period_labels: list[str],
    show_trend: bool,
    show_anomaly: bool,
    locations: list[str] = None,
    period_type: str = "monthly",
) -> go.Figure:
    """Create precipitation analysis plot."""
    fig = go.Figure()

    if locations is None:
        locations = ["All Stations"]

    colors = [
        "#1a5fb4",
        "#e74c3c",
        "#27ae60",
        "#2980b9",
        "#8e44ad",
        "#f39c12",
        "#d35400",
        "#16a085",
    ]

    prefix = (
        "Monthly"
        if period_type == "monthly"
        else ("Seasonal" if period_type == "seasonally" else "Yearly")
    )
    title_text = f"{prefix} Precipitation Analysis"

    for p_idx in range(1, len(period_labels) + 1):
        for i, loc in enumerate(locations):
            stats_df = _calculate_period_stats(
                merged_df, p_idx, "precipitation", location=loc
            )

            if stats_df is not None:
                stats_df = _add_anomaly_columns(stats_df)

            x = stats_df["year"].to_list() if stats_df is not None else []
            y = stats_df["avg"].to_list() if stats_df is not None else []
            anom_list = (
                stats_df["anomaly"].to_list() if stats_df is not None else []
            )
            c_data = (
                stats_df[
                    ["p25", "p75", "min", "max", "anomaly", "median", "trend"]
                ].rows()
                if stats_df is not None
                else []
            )

            color = colors[i % len(colors)]
            loc_prefix = f"{loc.split(',')[0]} - " if len(locations) > 1 else ""

            t_trace = _create_trend_trace(x, y, p_idx, show_trend)
            t_trace.name = f"{loc_prefix}{t_trace.name}"
            t_trace.line.color = color
            if len(locations) > 1:
                t_trace.line.dash = "dot"
            fig.add_trace(t_trace)

            if show_anomaly and len(locations) == 1:
                m_color = [a if a is not None else 0 for a in anom_list]
                m_cscale, show_colorbar = "BrBG", True
            else:
                m_color, m_cscale, show_colorbar = color, None, False

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    customdata=c_data,
                    mode="markers+lines",
                    name=f"{loc_prefix}Observations",
                    visible=(p_idx == 1),
                    marker=dict(
                        size=8 if len(locations) > 1 else 10,
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
                        if show_colorbar
                        else None,
                    ),
                    line=dict(width=1, color="rgba(0,0,0,0.2)"),
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{loc}</b><br><b>Year: %{{x}}</b><br>"
                        f"Total: %{{y:.1f}} mm<br>"
                        f"Median: %{{customdata[5]:.1f}} mm<br>"
                        f"Trend Mean: %{{customdata[6]:.1f}} mm<br>"
                        f"Mean Anomaly: %{{customdata[4]:.1f}} mm<br>"
                        f"Minimum: %{{customdata[2]:.1f}} mm<br>"
                        f"Maximum: %{{customdata[3]:.1f}} mm<br>"
                        f"25th Percentile: %{{customdata[0]:.1f}} mm<br>"
                        f"75th Percentile: %{{customdata[1]:.1f}} mm<br><extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, y=0.96, xanchor="center"),
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
    max_year = daily_df["year"].max()
    range_df = daily_df.group_by("station_id").agg(
        [
            pl.col("year").min().alias("min_y"),
            pl.col("year").max().alias("max_y"),
        ]
    )
    map_stations = stations_df.join(
        range_df, left_on="id", right_on="station_id"
    )
    map_stations = map_stations.with_columns(
        is_current=pl.col("max_y") == max_year
    )
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

    fig.update_layout(
        title=dict(
            text="Climate Station Locations", x=0.5, y=0.96, xanchor="center"
        ),
        map=dict(
            style="carto-positron",
            center=dict(
                lat=map_stations["latitude"].mean(),
                lon=map_stations["longitude"].mean(),
            ),
            zoom=8,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=500,
    )
    return fig
