"""
Climate analysis application with SQLite and HTTP caching.
Downloads daily climate data from MSC GeoMet and generates clean reports.
"""

from __future__ import annotations

import datetime
import logging
import math
import sqlite3
import sys
from typing import Any
from typing import Final
from typing import Iterable

import click
import jinja2
import plotly.graph_objects as go
import polars as pl
import requests_cache
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# Constants
API_BASE_URL: Final[str] = "https://api.weather.gc.ca"
COLLECTION_STATIONS: Final[str] = "climate-stations"
COLLECTION_DAILY: Final[str] = "climate-daily"
MAX_LIMIT: Final[int] = 10000
CACHE_DB: Final[str] = "climate_cache.sq3"
HTTP_CACHE_DB: Final[str] = "http_cache"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DAILY_SCHEMA: Final[dict[str, Any]] = {
    "station_id": pl.String,
    "date": pl.String,
    "year": pl.Int64,
    "month": pl.Int64,
    "day": pl.Int64,
    "temp_mean": pl.Float64,
    "temp_min": pl.Float64,
    "temp_max": pl.Float64,
    "precip": pl.Float64,
}


class ClimateCache:
    """Manages SQLite caching for daily climate data."""

    def __init__(self, db_path: str = CACHE_DB) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stations (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    latitude REAL,
                    longitude REAL,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_data (
                    station_id TEXT,
                    date TEXT,
                    year INTEGER,
                    month INTEGER,
                    day INTEGER,
                    temp_mean REAL,
                    temp_min REAL,
                    temp_max REAL,
                    precip REAL,
                    PRIMARY KEY (station_id, date)
                )
                """
            )

    def save_stations(self, df: pl.DataFrame) -> None:
        """Save station metadata to cache."""
        with sqlite3.connect(self.db_path) as conn:
            for row in df.iter_rows(named=True):
                conn.execute(
                    """
                    INSERT INTO stations (id, name, latitude, longitude)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        latitude=excluded.latitude,
                        longitude=excluded.longitude,
                        last_seen=CURRENT_TIMESTAMP
                    """,
                    (row["id"], row["name"], row["latitude"], row["longitude"]),
                )

    def get_cached_daily(
        self,
        station_id: str,
        start_year: int,
        end_year: int,
    ) -> pl.DataFrame:
        """Retrieve cached daily records for a station."""
        query = """
            SELECT station_id, date, year, month, day,
                   temp_mean, temp_min, temp_max, precip
            FROM daily_data
            WHERE station_id = ? AND year BETWEEN ? AND ?
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (station_id, start_year, end_year))
            rows = cursor.fetchall()
            if not rows:
                return pl.DataFrame(schema=DAILY_SCHEMA)

            data = []
            for r in rows:
                data.append(
                    {
                        "station_id": r[0],
                        "date": r[1],
                        "year": r[2],
                        "month": r[3],
                        "day": r[4],
                        "temp_mean": r[5],
                        "temp_min": r[6],
                        "temp_max": r[7],
                        "precip": r[8],
                    }
                )
            return pl.from_dicts(data, schema=DAILY_SCHEMA)

    def save_daily(self, df: pl.DataFrame) -> None:
        """Save daily observations to cache."""
        if df.is_empty():
            return
        with sqlite3.connect(self.db_path) as conn:
            for row in df.iter_rows(named=True):
                conn.execute(
                    """
                    INSERT INTO daily_data
                    (station_id, date, year, month, day,
                     temp_mean, temp_min, temp_max, precip)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(station_id, date) DO UPDATE SET
                        temp_mean=excluded.temp_mean,
                        temp_min=excluded.temp_min,
                        temp_max=excluded.temp_max,
                        precip=excluded.precip
                    """,
                    (
                        row["station_id"],
                        row["date"],
                        row["year"],
                        row["month"],
                        row["day"],
                        row["temp_mean"],
                        row["temp_min"],
                        row["temp_max"],
                        row["precip"],
                    ),
                )


class MSCClient:
    """Client for the MSC GeoMet API with caching."""

    def __init__(self, cache: ClimateCache) -> None:
        self.session = requests_cache.CachedSession(
            HTTP_CACHE_DB,
            backend="sqlite",
            expire_after=datetime.timedelta(days=7),
        )
        self.cache = cache

    def get_coordinates(self, location: str) -> tuple[float, float] | None:
        """Get coordinates for a location string."""
        geolocator = Nominatim(user_agent="climate_analysis_tool")
        loc = geolocator.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
        return None

    def find_stations_near(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> pl.DataFrame:
        """Find stations within a radius of a point."""
        lat_buf = radius_km / 111.0
        lon_buf = radius_km / (111.0 * math.cos(math.radians(lat)))
        bbox = (
            f"{lon - lon_buf},{lat - lat_buf},{lon + lon_buf},{lat + lat_buf}"
        )

        url = f"{API_BASE_URL}/collections/{COLLECTION_STATIONS}/items"
        params: dict[str, Any] = {
            "f": "json",
            "bbox": bbox,
            "limit": 1000,
        }

        logger.info(f"Searching for stations near {lat}, {lon}...")
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        stations = []
        for feature in data.get("features", []):
            props = feature["properties"]
            s_lat = feature["geometry"]["coordinates"][1]
            s_lon = feature["geometry"]["coordinates"][0]
            dist = geodesic((lat, lon), (s_lat, s_lon)).km

            if dist <= radius_km:
                stations.append(
                    {
                        "id": props["CLIMATE_IDENTIFIER"],
                        "name": props["STATION_NAME"],
                        "latitude": s_lat,
                        "longitude": s_lon,
                        "distance_km": dist,
                    }
                )

        df = pl.DataFrame(stations)
        self.cache.save_stations(df)
        return df

    def fetch_daily_data(
        self,
        station_ids: Iterable[str],
        start_year: int,
        end_year: int,
    ) -> pl.DataFrame:
        """Fetch daily data and aggregate to monthly."""
        all_dfs = []
        url = f"{API_BASE_URL}/collections/{COLLECTION_DAILY}/items"

        for sid in station_ids:
            # 1. Check Custom Cache
            cached_df = self.cache.get_cached_daily(sid, start_year, end_year)

            # Check if we have data for all requested years
            if not cached_df.is_empty():
                years_found = cached_df["year"].unique().to_list()
                requested_years = list(range(start_year, end_year + 1))
                if all(y in years_found for y in requested_years):
                    logger.info(f"Using structured cache for station {sid}")
                    all_dfs.append(cached_df)
                    continue

            # 2. Download (HTTP cache handles redundant requests)
            logger.info(f"Fetching daily data for station {sid}...")
            station_data = []
            dt_filter = f"{start_year}-01-01/{end_year}-12-31"
            offset = 0
            while True:
                params = {
                    "f": "json",
                    "CLIMATE_IDENTIFIER": sid,
                    "datetime": dt_filter,
                    "limit": MAX_LIMIT,
                    "offset": offset,
                }
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                features = data.get("features", [])
                if not features:
                    break

                for f in features:
                    p = f["properties"]
                    station_data.append(
                        {
                            "station_id": p["CLIMATE_IDENTIFIER"],
                            "date": p["LOCAL_DATE"],
                            "year": int(p["LOCAL_YEAR"]),
                            "month": int(p["LOCAL_MONTH"]),
                            "day": int(p["LOCAL_DAY"]),
                            "temp_mean": p.get("MEAN_TEMPERATURE"),
                            "temp_min": p.get("MIN_TEMPERATURE"),
                            "temp_max": p.get("MAX_TEMPERATURE"),
                            "precip": p.get("TOTAL_PRECIPITATION"),
                        }
                    )

                returned = data.get("numberReturned", 0)
                matched = data.get("numberMatched", 0)
                offset += returned
                if offset >= matched or returned == 0:
                    break

            if not station_data:
                new_df = pl.DataFrame(schema=DAILY_SCHEMA)
            else:
                new_df = pl.from_dicts(station_data, schema=DAILY_SCHEMA)

            self.cache.save_daily(new_df)

            combined = (
                pl.concat([cached_df, new_df])
                .unique(subset=["station_id", "date"])
                .filter(
                    (pl.col("year") >= start_year)
                    & (pl.col("year") <= end_year)
                )
            )
            all_dfs.append(combined)

        if not all_dfs:
            return pl.DataFrame(schema=DAILY_SCHEMA)
        return pl.concat(all_dfs)


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


def calculate_trendline(x: list[float], y: list[float]) -> list[float] | None:
    """Calculate simple linear trendline."""
    if len(x) < 2:
        return None
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None

    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - m * sum_x) / n
    return [m * xi + b for xi in x]


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

    # Aggregate Daily -> Monthly per station
    monthly_df = daily_df.group_by(["station_id", "year", "month"]).agg(
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

    html_sections = []
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

    # Trace count: 0:Dev, 1:Range, 2:Trend, 3:Main
    traces_per_month = 4

    # 1. Temperature Chart
    fig_temp = go.Figure()
    for m_idx in range(1, 13):
        m_df = monthly_df.filter(pl.col("month") == m_idx).sort("year")

        stats = None
        if not m_df.is_empty():
            stats = (
                m_df.group_by("year")
                .agg(
                    [
                        pl.col("temp_mean").mean().alias("avg"),
                        pl.col("temp_q1").mean().alias("q1"),
                        pl.col("temp_q3").mean().alias("q3"),
                        pl.col("temp_min_abs").min().alias("min"),
                        pl.col("temp_max_abs").max().alias("max"),
                    ]
                )
                .sort("year")
            )
            lt_mean = stats["avg"].mean()
            stats = stats.with_columns(anomaly=pl.col("avg") - lt_mean)

        x = stats["year"].to_list() if stats is not None else []
        y = stats["avg"].to_list() if stats is not None else []
        anom_list = stats["anomaly"].to_list() if stats is not None else []
        c_data = (
            stats[["q1", "q3", "min", "max", "anomaly"]].rows()
            if stats is not None
            else []
        )

        # Trace 0: Shading (Trend-Relative)
        y_upper, y_lower = [], []
        if stats is not None:
            x_vals = [float(xi) for xi in x]
            trend_y = calculate_trendline(x_vals, y)
            if trend_y:
                resids = [yi - tyi for yi, tyi in zip(y, trend_y)]
                std_resid = (sum(r**2 for r in resids) / len(resids)) ** 0.5
                y_upper = [ty + std_resid for ty in trend_y]
                y_lower = [ty - std_resid for ty in trend_y]

        fig_temp.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor="rgba(200, 200, 200, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Std Dev (Trend-rel.)",
                visible=(m_idx == 1 and shade_deviation),
                showlegend=(m_idx == 1 and shade_deviation),
                hoverinfo="skip",
            )
        )

        # Trace 1: Q1/Q3 Range
        q3 = stats["q3"] if stats is not None else []
        q1 = stats["q1"] if stats is not None else []
        avg = stats["avg"] if stats is not None else []
        fig_temp.add_trace(
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
                showlegend=(m_idx == 1),
                hoverinfo="skip",
            )
        )

        # Trace 2: Trendline
        x_vals = [float(xi) for xi in x]
        trend_y = calculate_trendline(x_vals, y) if stats is not None else None
        fig_temp.add_trace(
            go.Scatter(
                x=x,
                y=trend_y if trend_y else [],
                mode="lines",
                name="Linear Trend",
                visible=(m_idx == 1 and show_trend),
                line=dict(width=2.5, color="#e74c3c", dash="dash"),
                showlegend=(m_idx == 1 and show_trend),
                hoverinfo="skip",
            )
        )

        # Trace 3: Main Data
        m_color = anom_list if show_anomaly else "#2c3e50"
        m_cscale = "RdBu_r" if show_anomaly else None
        fig_temp.add_trace(
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
                    colorbar=(
                        dict(
                            title=dict(text="Anomaly (°C)", side="top"),
                            orientation="h",
                            x=0.5,
                            y=-0.18,
                            yanchor="top",
                            xanchor="center",
                            thickness=15,
                            len=0.5,
                        )
                        if (m_idx == 1 and show_anomaly)
                        else None
                    ),
                ),
                line=dict(width=1, color="rgba(0,0,0,0.2)"),
                showlegend=(m_idx == 1),
                hovertemplate=(
                    "<b>Year: %{x}</b><br>Mean: %{y:.1f}°C<br>"
                    "Anomaly: %{customdata[4]:.1f}°C "
                    "(%{customdata[5]:.1f}%)<br>"
                    "Q1: %{customdata[0]:.1f}°C<br>"
                    "Q3: %{customdata[1]:.1f}°C<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig_temp.update_layout(
        title=dict(
            text="Monthly Temperature Analysis", x=0.5, y=0.96, xanchor="center"
        ),
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02
        ),
        margin=dict(l=60, r=160, t=80, b=100),
        dragmode="pan",
        height=600,
    )
    create_modern_theme(fig_temp)
    html_sections.append(
        fig_temp.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            div_id="chart-temp",
            config={"modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        )
    )

    # 2. Precipitation Chart
    fig_pr = go.Figure()
    for m_idx in range(1, 13):
        m_df = monthly_df.filter(pl.col("month") == m_idx).sort("year")

        stats = None
        if not m_df.is_empty():
            stats = (
                m_df.group_by("year")
                .agg(
                    [
                        pl.col("precip_total").mean().alias("avg"),
                        pl.col("precip_q1").mean().alias("q1"),
                        pl.col("precip_q3").mean().alias("q3"),
                        pl.col("precip_total").min().alias("min"),
                        pl.col("precip_total").max().alias("max"),
                    ]
                )
                .sort("year")
            )
            lt_mean = stats["avg"].mean()
            stats = stats.with_columns(anomaly=pl.col("avg") - lt_mean)

        x = stats["year"].to_list() if stats is not None else []
        y = stats["avg"].to_list() if stats is not None else []
        anom_list = stats["anomaly"].to_list() if stats is not None else []
        c_data = (
            stats[["q1", "q3", "min", "max", "anomaly"]].rows()
            if stats is not None
            else []
        )

        # Trace 0: Shading
        y_upper, y_lower = [], []
        if stats is not None:
            x_vals = [float(xi) for xi in x]
            trend_y = calculate_trendline(x_vals, y)
            if trend_y:
                resids = [yi - tyi for yi, tyi in zip(y, trend_y)]
                std_resid = (sum(r**2 for r in resids) / len(resids)) ** 0.5
                y_upper = [ty + std_resid for ty in trend_y]
                y_lower = [ty - std_resid for ty in trend_y]

        fig_pr.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor="rgba(100, 150, 200, 0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Std Dev (Trend-rel.)",
                visible=(m_idx == 1 and shade_deviation),
                showlegend=(m_idx == 1 and shade_deviation),
                hoverinfo="skip",
            )
        )

        # Trace 1: Range (Removed for Precipitation)
        fig_pr.add_trace(
            go.Scatter(x=[], y=[], visible=False, showlegend=False)
        )

        # Trace 2: Trend
        x_vals = [float(xi) for xi in x]
        trend_y = calculate_trendline(x_vals, y) if stats is not None else None
        fig_pr.add_trace(
            go.Scatter(
                x=x,
                y=trend_y if trend_y else [],
                mode="lines",
                name="Linear Trend",
                visible=(m_idx == 1 and show_trend),
                line=dict(width=2.5, color="#e74c3c", dash="dash"),
                showlegend=(m_idx == 1 and show_trend),
                hoverinfo="skip",
            )
        )

        # Trace 3: Main
        m_color = anom_list if show_anomaly else "#1a5fb4"
        m_cscale = "BrBG" if show_anomaly else None
        fig_pr.add_trace(
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
                    colorbar=(
                        dict(
                            title=dict(text="Anomaly (mm)", side="top"),
                            orientation="h",
                            x=0.5,
                            y=-0.18,
                            yanchor="top",
                            xanchor="center",
                            thickness=15,
                            len=0.5,
                        )
                        if (m_idx == 1 and show_anomaly)
                        else None
                    ),
                ),
                line=dict(width=1, color="rgba(0,0,0,0.2)"),
                showlegend=(m_idx == 1),
                hovertemplate=(
                    "<b>Year: %{x}</b><br>Total: %{y:.1f} mm<br>"
                    "Anomaly: %{customdata[4]:.1f} mm "
                    "(%{customdata[5]:.1f}%)<br>"
                    "Q1: %{customdata[0]:.1f} mm<br>"
                    "Q3: %{customdata[1]:.1f} mm<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig_pr.update_layout(
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
    create_modern_theme(fig_pr)
    html_sections.append(
        fig_pr.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="chart-precip",
            config={"modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        )
    )

    # 3. Map
    fig_map = go.Figure()
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
        fig_map.add_trace(
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
        fig_map.update_layout(
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
        html_sections.append(
            fig_map.to_html(
                full_html=False, include_plotlyjs=False, div_id="chart-map"
            )
        )

    f_url = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"><title>Climate Analysis - {{ location }}</title>
        <link href="{{ f_url }}" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 2em;
                background: #f8f9fa;
                color: #333;
            }
            .card {
                background: white;
                padding: 2em;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                margin-bottom: 2em;
            }
            h1 { font-weight: 600; text-align: center; margin: 0; }
            header {
                margin-bottom: 2.5em;
                border-bottom: 1px solid #ddd;
                padding-bottom: 1.5em;
            }
            .meta {
                color: #666;
                text-align: center;
                margin-top: 0.5em;
                font-size: 0.9em;
            }
            .controls {
                background: #fff;
                padding: 1em;
                border-radius: 8px;
                margin-bottom: 2em;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            }
            select {
                padding: 8px 16px;
                border-radius: 6px;
                border: 1px solid #ccc;
                font-family: inherit;
                cursor: pointer;
            }
            footer {
                text-align: center;
                color: #999;
                font-size: 0.8em;
                margin-top: 2em;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Climate Analysis: {{ location }}</h1>
            <div class="meta">
                {{ radius }}km radius. Aggregated Daily Data.<br>
                Generated: {{ date }}
            </div>
        </header>
        <div class="controls">
            <label for="month-select"><b>Select Month:</b></label>
            <select id="month-select">
                {% for m in months %}
                    <option value="{{ loop.index0 }}">{{ m }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="card">{{ plots[0] }}</div>
        <div class="card">{{ plots[1] }}</div>
        {% if plots|length > 2 %}
            <div class="card">{{ plots[2] }}</div>
        {% endif %}
        <footer>Climate Analysis Tool &copy; 2026</footer>
        <script>
        const tracesPerMonth = {{ traces_per_month }};
        const showTrend = {{ show_trend|tojson }};
        const showDev = {{ show_dev|tojson }};
        document.getElementById('month-select')
            .addEventListener('change', function(e) {
            const mIdx = parseInt(e.target.value);
            ['chart-temp', 'chart-precip'].forEach(id => {
                const gd = document.getElementById(id);
                if (!gd) return;
                const vis = Array.from(
                    { length: 12 * tracesPerMonth },
                    (_, i) => {
                        const month = Math.floor(i / tracesPerMonth);
                        const type = i % tracesPerMonth;
                        if (month !== mIdx) return false;
                        if (type === 0) return showDev;
                        if (type === 2) return showTrend;
                        return true;
                    }
                );
                Plotly.restyle(gd, {visible: vis});
            });
        });
        </script>
    </body></html>
    """
    return jinja2.Template(template).render(
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


@click.command()
@click.option("--location", required=True, help="Location name")
@click.option("--radius", default=100.0, help="Radius (km)")
@click.option("--start-year", default=1900, help="Start year")
@click.option("--end_year", default=None, help="End year")
@click.option("--trend", is_flag=True, help="Show trendlines")
@click.option("--shade-deviation", is_flag=True, help="Shade deviation")
@click.option("--no-anomaly", is_flag=True, help="Disable anomaly coloring")
def main(
    location: str,
    radius: float,
    start_year: int,
    end_year: int | None,
    trend: bool = False,
    shade_deviation: bool = False,
    no_anomaly: bool = False,
) -> None:
    if end_year is None:
        end_year = datetime.datetime.now().year
    cache = ClimateCache()
    client = MSCClient(cache)
    coords = client.get_coordinates(location)
    if not coords:
        logger.error(f"Failed to find {location}")
        sys.exit(1)
    lat, lon = coords
    logger.info(f"Location: {location} ({lat}, {lon})")
    stations_df = client.find_stations_near(lat, lon, radius)
    if stations_df.is_empty():
        logger.error("No stations found.")
        sys.exit(1)
    daily_df = client.fetch_daily_data(
        stations_df["id"].to_list(), start_year, end_year
    )
    if daily_df.is_empty():
        logger.error("No data found.")
        sys.exit(1)
    # Determine if anomaly coloring should be shown:
    # Only if shade_deviation is requested AND not explicitly disabled
    show_anomaly_plot = shade_deviation and not no_anomaly

    report = generate_report(
        daily_df,
        stations_df,
        location,
        radius,
        trend,
        shade_deviation,
        show_anomaly_plot,
    )
    fname = f"climate_report_{location.lower().replace(' ', '_')}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report: {fname}")


if __name__ == "__main__":
    main()
