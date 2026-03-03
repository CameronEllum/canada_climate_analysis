"""
Climate analysis application with SQLite and HTTP caching.
Downloads daily climate data from MSC GeoMet and generates clean reports.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Annotated
from typing import Optional

import typer

from climate_app import ClimateApp
from config import AggregateMode
from config import ProcessingConfig

# Logging configuration
DEFAULT_LOG_LEVEL = "INFO"
log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

logging.basicConfig(
    level=log_level,
    format="%(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Climate analysis application.")

# CLI Option Aliases to avoid "Vertical Wall"
LocationOpt = Annotated[list[str], typer.Option("--location", help="Location name")]
RadiusOpt = Annotated[float, typer.Option(help="Radius (km)")]
StartYearOpt = Annotated[int, typer.Option(help="Start year")]
EndYearOpt = Annotated[Optional[int], typer.Option(help="End year")]
TrendOpt = Annotated[bool, typer.Option(help="Show trendlines")]
MedianOpt = Annotated[bool, typer.Option(help="Show median line")]
AnomalyOpt = Annotated[bool, typer.Option(help="Show anomaly plot (default: True)")]
MaxTempOpt = Annotated[
    bool,
    typer.Option(
        "--max/--no-max", help="Use maximum daily temperature (default: False)"
    ),
]
MinTempOpt = Annotated[
    bool,
    typer.Option(
        "--min/--no-min", help="Use minimum daily temperature (default: False)"
    ),
]
AggregateOpt = Annotated[
    AggregateMode,
    typer.Option(
        "--mode",
        case_sensitive=False,
        help="Aggregation mode (monthly, seasonally, or yearly)",
    ),
]
CachePathOpt = Annotated[
    str, typer.Option("--cache", help="Path to SQLite cache (default: climate_cache.sq3)")
]
CacheRequestsOpt = Annotated[
    bool,
    typer.Option(
        "--cache-requests/--no-cache-requests",
        help="Enable HTTP requests caching (default: False)",
    ),
]
PercentilesOpt = Annotated[
    list[int],
    typer.Option(
        "--percentiles",
        help="Percentile integers for ribbons (specify multiple times, e.g., --percentiles 10 --percentiles 20)",
    ),
]
CacheReportOpt = Annotated[
    bool, typer.Option("--cache-report", help="Generate a report of the data in cache.")
]


@app.command()
def main(
    location: LocationOpt = [],
    radius: RadiusOpt = 100.0,
    start_year: StartYearOpt = 1900,
    end_year: EndYearOpt = None,
    trend: TrendOpt = True,
    median: MedianOpt = False,
    show_anomaly: AnomalyOpt = True,
    max_temp: MaxTempOpt = False,
    min_temp: MinTempOpt = False,
    mode: AggregateOpt = AggregateMode.monthly,
    cache_path: CachePathOpt = "climate_cache.sq3",
    cache_requests: CacheRequestsOpt = False,
    percentiles: PercentilesOpt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    cache_report: CacheReportOpt = False,
) -> None:
    """Climate Analysis Tool: Download data and generate reports."""
    climate_app = ClimateApp(cache_path, cache_requests=cache_requests)

    if cache_report:
        climate_app.generate_cache_report()
        return

    # Process and validate args into a config object
    config = ProcessingConfig.from_args(
        location=location,
        radius=radius,
        start_year=start_year,
        end_year=end_year,
        trend=trend,
        median=median,
        show_anomaly=show_anomaly,
        max_temp=max_temp,
        min_temp=min_temp,
        mode=mode,
        percentiles=percentiles,
    )

    try:
        fpath = climate_app.run_analysis(config)
        logger.info(f"Report generated: {fpath}")
    except RuntimeError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
