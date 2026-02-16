"""
Climate analysis application with SQLite and HTTP caching.
Downloads daily climate data from MSC GeoMet and generates clean reports.
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

import click
import polars as pl

from climate_cache import ClimateCache
from msc_client import MSCClient
from report_generator import generate_report

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--location", required=True, multiple=True, help="Location name")
@click.option("--radius", default=100.0, help="Radius (km)")
@click.option("--start-year", default=1900, type=int, help="Start year")
@click.option("--end-year", default=None, type=int, help="End year")
@click.option("--trend/--no-trend", default=True, help="Show trendlines")
@click.option(
    "--std-dev/--no-std-dev", default=False, help="Shade standard deviation"
)
@click.option(
    "--show-anomaly/--no-anomaly",
    default=True,
    help="Show anomaly plot (default: True)",
)
@click.option(
    "--max/--no-max",
    "max_temp",
    default=False,
    help="Use maximum daily temperature (default: False)",
)
@click.option(
    "--min/--no-min",
    "min_temp",
    default=False,
    help="Use minimum daily temperature (default: False)",
)
@click.option(
    "--monthly",
    "period",
    flag_value="monthly",
    default=True,
    help="Aggregate by month (default)",
)
@click.option(
    "--seasonally",
    "period",
    flag_value="seasonally",
    help="Aggregate by season",
)
@click.option(
    "--yearly", "period", flag_value="yearly", help="Aggregate by year"
)
@click.option(
    "--cache",
    "cache_path",
    default="climate_cache.sq3",
    help="Path to SQLite cache (default: climate_cache.sq3)",
)
@click.option(
    "--cache-requests/--no-cache-requests",
    default=False,
    help="Enable HTTP requests caching (default: False)",
)
def main(
    location: tuple[str, ...],
    radius: float,
    start_year: int,
    end_year: int,
    trend: bool,
    std_dev: bool,
    show_anomaly: bool,
    max_temp: bool,
    min_temp: bool,
    period: str,
    cache_path: str,
    cache_requests: bool,
) -> None:
    """Generate climate analysis report for multiple locations."""
    if end_year is None:
        end_year = datetime.datetime.now().year

    cache = ClimateCache(cache_path)
    client = MSCClient(cache, cache_requests=cache_requests)

    all_stations_dfs = []

    normalized_locations = []
    for loc in location:
        # Normalize: ensure it ends with ",Canada"
        norm_loc = loc if loc.endswith(",Canada") else f"{loc},Canada"
        normalized_locations.append(norm_loc)

    # Deduplicate while preserving order
    normalized_locations = list(dict.fromkeys(normalized_locations))

    for norm_loc in normalized_locations:
        coords = client.get_coordinates(norm_loc)
        if not coords:
            logger.warning(f"Failed to find {norm_loc}, skipping...")
            continue

        lat, lon = coords
        logger.info(f"Location: {norm_loc} ({lat}, {lon})")

        stations_df = client.find_stations_near(lat, lon, radius)
        if not stations_df.is_empty():
            # Add location tag to stations for report labeling
            stations_df = stations_df.with_columns(
                pl.lit(norm_loc).alias("requested_location")
            )
            all_stations_dfs.append(stations_df)

    if not all_stations_dfs:
        logger.error("No stations found for any specified location.")
        sys.exit(1)

    # Combine and unique stations
    combined_stations_df = pl.concat(all_stations_dfs).unique(subset=["id"])

    daily_df = client.fetch_daily_data(
        combined_stations_df["id"].to_list(), start_year, end_year
    )
    if daily_df.is_empty():
        logger.error("No data found.")
        sys.exit(1)

    # Determine if anomaly coloring should be shown
    # Respect the user's explicit choice via --show-anomaly/--no-anomaly

    if len(normalized_locations) > 1 and show_anomaly:
        logger.warning(
            "Anomaly plots are disabled when multiple locations are requested."
        )
        show_anomaly = False

    report = generate_report(
        daily_df,
        combined_stations_df,
        normalized_locations,
        radius,
        trend,
        std_dev,
        show_anomaly,
        max_temp,
        min_temp,
        period=period,
    )

    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Determine year range for filename
    effective_end_year = end_year or daily_df["year"].max()
    year_range = f"{start_year}-{effective_end_year}"

    # Filename based on period, year range, and location(s)
    loc_part = normalized_locations[0].split(",")[0].lower().replace(" ", "_")
    if len(normalized_locations) > 1:
        loc_part += "_and_others"

    fname = f"climate_report_{period}_{year_range}_{loc_part}.html"
    fpath = reports_dir / fname

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report: {fpath}")


if __name__ == "__main__":
    main()
