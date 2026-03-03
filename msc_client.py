"""MSC GeoMet API client with caching."""

from __future__ import annotations

import datetime
import logging
import math
from typing import Any
from typing import Final

import polars as pl
import requests
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

# Schema for stations table
STATION_SCHEMA = {
    "id": pl.String,
    "name": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "distance_km": pl.Float64,
}

logger = logging.getLogger(__name__)


class MSCClient:
    """Client for the MSC GeoMet API with caching."""

    def __init__(
        self, cache: ClimateCache, cache_requests: bool = False
    ) -> None:
        if cache_requests:
            self.session = requests_cache.CachedSession(
                HTTP_CACHE_DB,
                backend="sqlite",
                expire_after=datetime.timedelta(days=7),
            )
        else:
            self.session = requests.Session()
        self.cache = cache

    def get_coordinates(self, location: str) -> tuple[float, float] | None:
        """Get coordinates for a location string with retry logic."""
        geolocator = Nominatim(user_agent="climate_analysis_tool")

        for attempt in range(3):
            try:
                loc = geolocator.geocode(location, timeout=10)
                if loc:
                    return float(loc.latitude), float(loc.longitude)
                return None
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Geocoding failed for {location}: {e}")
                    return None
                logger.warning(
                    f"Geocoding attempt {attempt + 1} failed, retrying..."
                )
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
        try:
            logger.info(f"Sending API request to {COLLECTION_STATIONS}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            logger.info("Received station list from API.")
            data = response.json()
            features = data.get("features", [])
            logger.info(
                f"API returned {len(features)} candidate stations. Filtering by distance..."
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch stations: {e}")
            return pl.DataFrame(schema=STATION_SCHEMA)

        stations = []
        for feature in data.get("features", []):
            try:
                props = feature["properties"]
                geom = feature["geometry"]
                s_lat = float(geom["coordinates"][1])
                s_lon = float(geom["coordinates"][0])
                dist = geodesic((lat, lon), (s_lat, s_lon)).km

                if dist <= radius_km:
                    stations.append(
                        {
                            "id": str(props["CLIMATE_IDENTIFIER"]),
                            "name": str(props["STATION_NAME"]),
                            "latitude": s_lat,
                            "longitude": s_lon,
                            "distance_km": float(dist),
                        }
                    )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed station feature: {e}")
                continue

        if not stations:
            return pl.DataFrame(schema=STATION_SCHEMA)

        df = pl.DataFrame(stations)
        self.cache.save_stations(df)
        return df

    def fetch_daily_data(
        self,
        station_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> pl.DataFrame:
        """Fetch daily data for stations, using cache to fill gaps."""
        missing_blocks = self.cache.get_missing_blocks(
            station_ids, start_year, end_year
        )

        if missing_blocks:
            num_blocks = len(missing_blocks)
            logger.info(
                f"Found {num_blocks} missing data blocks to download. Fetching now..."
            )
            for i, (sid, start_date, end_date) in enumerate(missing_blocks):
                logger.debug(
                    f"Block {i + 1}/{num_blocks}: Downloading station {sid} for period {start_date} to {end_date}"
                )
                if i > 0 and i % 5 == 0:
                    logger.info(
                        f"Download Progress: {i}/{num_blocks} blocks completed..."
                    )
                self._fetch_and_cache_block(sid, start_date, end_date)
            logger.info("Data download complete.")
        else:
            logger.info("All requested data blocks found in cache.")

        # Final retrieval from cache
        return self.cache.get_daily_data(station_ids, start_year, end_year)

    def _fetch_and_cache_block(
        self, station_id: str, start_date: str, end_date: str
    ) -> None:
        """Fetch a specific block of data and save to cache."""
        logger.debug(
            f"Fetching data for station {station_id} ({start_date} to {end_date})"
        )
        url = f"{API_BASE_URL}/collections/{COLLECTION_DAILY}/items"

        params: dict[str, Any] = {
            "f": "json",
            "CLIMATE_IDENTIFIER": station_id,
            "datetime": f"{start_date}/{end_date}",
            "limit": MAX_LIMIT,
        }

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(
                f"API request failed for {station_id} ({start_date} to {end_date}): {e}"
            )
            return

        records = []
        for feature in data.get("features", []):
            try:
                p = feature["properties"]

                def _to_float(val: Any) -> float | None:
                    try:
                        return float(val) if val is not None else None
                    except ValueError, TypeError:
                        return None

                records.append(
                    {
                        "station_id": str(p["CLIMATE_IDENTIFIER"]),
                        "date": str(p["LOCAL_DATE"]),
                        "temp_max": _to_float(p.get("MAX_TEMPERATURE")),
                        "temp_min": _to_float(p.get("MIN_TEMPERATURE")),
                        "temp_mean": _to_float(p.get("MEAN_TEMPERATURE")),
                        "precip_total": _to_float(p.get("TOTAL_PRECIPITATION")),
                    }
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Malformed record in API response: {e}")
                continue

        df = (
            pl.DataFrame(records, schema=DAILY_SCHEMA)
            if records
            else pl.DataFrame(schema=DAILY_SCHEMA)
        )
        self.cache.save_daily_request(station_id, start_date, end_date, df)
