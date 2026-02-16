"""MSC GeoMet API client with caching."""

from __future__ import annotations

import datetime
import logging
import math
from typing import Any
from typing import Final
from typing import Iterable

import polars as pl
import requests_cache
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from climate_cache import DAILY_SCHEMA
from climate_cache import ClimateCache

# Constants
API_BASE_URL: Final[str] = "https://api.weather.gc.ca"
COLLECTION_STATIONS: Final[str] = "climate-stations"
COLLECTION_DAILY: Final[str] = "climate-daily"
MAX_LIMIT: Final[int] = 10000
HTTP_CACHE_DB: Final[str] = "http_cache"

logger = logging.getLogger(__name__)


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
