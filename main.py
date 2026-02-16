"""
Climate analysis application with SQLite and HTTP caching.
Downloads daily climate data from MSC GeoMet and generates clean reports.
"""

from __future__ import annotations

import datetime
import logging
import sys

import click

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
@click.option("--location", required=True, help="Location name")
@click.option("--radius", default=100.0, help="Radius (km)")
@click.option("--start-year", default=1900, help="Start year")
@click.option("--end_year", default=None, help="End year")
@click.option("--trend/--no-trend", default=True, help="Show trendlines")
@click.option("--shade-deviation", is_flag=True, help="Shade deviation")
@click.option("--no-anomaly", is_flag=True, help="Disable anomaly coloring")
def main(
    location: str,
    radius: float,
    start_year: int,
    end_year: int | None,
    trend: bool = True,
    shade_deviation: bool = False,
    no_anomaly: bool = False,
) -> None:
    """Generate climate analysis report for a location."""
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

    # Determine if anomaly coloring should be shown
    show_anomaly_plot = shade_deviation and not no_anomaly

    report = generate_report(
        daily_df,
        stations_df,
        location,
        coords,
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
