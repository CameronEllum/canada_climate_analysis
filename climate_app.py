"""Main orchestration logic for the climate analysis application."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import polars as pl

from climate_cache import ClimateCache
from msc_client import MSCClient
from report_generator import generate_report
from config import ProcessingConfig

logger = logging.getLogger(__name__)


class ClimateApp:
    """Orchestrates climate data fetching and report generation."""

    def __init__(
        self,
        cache_path: str = "climate_cache.sq3",
        cache_requests: bool = False,
    ):
        self.cache = ClimateCache(cache_path)
        self.client = MSCClient(self.cache, cache_requests=cache_requests)

    def run_analysis(self, config: ProcessingConfig) -> Path:
        """Run the full analysis pipeline and return the path to the report."""
        effective_end_year = config.end_year
        if effective_end_year is None:
            effective_end_year = datetime.datetime.now().year

        normalized_locations = []
        for loc in config.locations:
            norm_loc = loc if loc.endswith(",Canada") else f"{loc},Canada"
            normalized_locations.append(norm_loc)

        # Deduplicate while preserving order
        normalized_locations = list(dict.fromkeys(normalized_locations))

        all_stations_dfs = []
        for norm_loc in normalized_locations:
            coords = self.client.get_coordinates(norm_loc)
            if not coords:
                logger.warning(f"Failed to find {norm_loc}, skipping...")
                continue

            lat, lon = coords
            logger.info(f"Location: {norm_loc} ({lat}, {lon})")

            stations_df = self.client.find_stations_near(lat, lon, config.radius)
            if not stations_df.is_empty():
                stations_df = stations_df.with_columns(
                    pl.lit(norm_loc).alias("requested_location")
                )
                all_stations_dfs.append(stations_df)

        if not all_stations_dfs:
            raise RuntimeError("No stations found for any specified location.")

        combined_stations_df = pl.concat(all_stations_dfs).unique(subset=["id"])

        daily_df = self.client.fetch_daily_data(
            combined_stations_df["id"].to_list(),
            config.start_year,
            effective_end_year,
        )
        if daily_df.is_empty():
            raise RuntimeError("No data found.")

        show_anomaly = config.show_anomaly
        if len(normalized_locations) > 1 and show_anomaly:
            logger.warning(
                "Anomaly plots are disabled when multiple locations are requested."
            )
            show_anomaly = False

        report_html = generate_report(
            daily_df,
            combined_stations_df,
            normalized_locations,
            config.radius,
            config.trend,
            config.median,
            show_anomaly,
            config.max_temp,
            config.min_temp,
            period=config.period,
            ribbon_percentiles=config.percentiles,
        )

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        effective_report_end = effective_end_year or daily_df["year"].max()
        year_range = f"{config.start_year}-{effective_report_end}"
        loc_part = normalized_locations[0].split(",")[0].lower().replace(" ", "_")
        if len(normalized_locations) > 1:
            loc_part += "_and_others"

        fname = f"climate_report_{config.period}_{year_range}_{loc_part}.html"
        fpath = reports_dir / fname

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(report_html)

        return fpath

    def generate_cache_report(self) -> None:
        """Generate and print a report of the data held in the cache."""
        summary = self.cache.get_cache_summary()
        if summary.is_empty():
            print("Cache is empty.")
            return

        print("\n=== Climate Data Cache Report ===")
        print(f"Total Stations Cached: {len(summary)}")
        print("-" * 50)

        # Format for display
        with pl.Config(tbl_rows=100, tbl_width_chars=120):
            print(
                summary.select(
                    [
                        pl.col("station_id").alias("ID"),
                        pl.col("name").alias("Station Name"),
                        pl.col("start_date").alias("Start"),
                        pl.col("end_date").alias("End"),
                        pl.col("days_cached").alias("Days"),
                    ]
                )
            )
        print("-" * 50)

        total_days = summary["days_cached"].sum()
        print(f"Total Observations: {total_days:,} days")
        print("==============================\n")
