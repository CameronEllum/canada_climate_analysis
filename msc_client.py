"""MSC GeoMet API client with caching."""

from __future__ import annotations

import datetime
import logging
import math
from typing import Any
from typing import Final
from typing import Iterable

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

    def _get_missing_blocks(
        self,
        sid: str,
        req_start: datetime.date,
        req_end: datetime.date,
    ) -> list[tuple[datetime.date, datetime.date]]:
        """Identify missing date blocks for a station."""
        existing_dates = self.cache.get_existing_dates(
            sid, req_start.isoformat(), req_end.isoformat()
        )
        checked_ranges = self.cache.get_station_requests(sid)

        # Generate all requested dates
        all_req_dates = []
        curr = req_start
        while curr <= req_end:
            all_req_dates.append(curr.isoformat())
            curr += datetime.date.resolution

        # Subtract existing and "empty" ranges
        missing_dates = []
        for d_str in all_req_dates:
            if d_str in existing_dates:
                continue
            is_checked = False
            for s, e, _status in checked_ranges:
                if s <= d_str <= e:
                    is_checked = True
                    break
            if not is_checked:
                missing_dates.append(d_str)

        # Group into blocks
        blocks = []
        if missing_dates:
            missing_dates.sort()
            b_start = datetime.date.fromisoformat(missing_dates[0])
            prev = b_start
            for i in range(1, len(missing_dates)):
                curr_d = datetime.date.fromisoformat(missing_dates[i])
                if (curr_d - prev).days > 1:
                    blocks.append((b_start, prev))
                    b_start = curr_d
                prev = curr_d
            blocks.append((b_start, prev))
        return blocks

    def _fetch_and_cache_block(
        self, sid: str, b_start: datetime.date, b_end: datetime.date
    ) -> None:
        """Fetch a single contiguous block from MSC and save to cache."""
        logger.info(f"[{sid}] Downloading new data: {b_start} to {b_end}...")
        url = f"{API_BASE_URL}/collections/{COLLECTION_DAILY}/items"
        station_data = []
        dt_filter = f"{b_start.isoformat()}/{b_end.isoformat()}"
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

        if station_data:
            new_df = pl.from_dicts(station_data, schema=DAILY_SCHEMA)
            self.cache.save_daily(new_df)
            self.cache.log_station_request(
                sid, b_start.isoformat(), b_end.isoformat(), "SUCCESS"
            )
        else:
            self.cache.log_station_request(
                sid, b_start.isoformat(), b_end.isoformat(), "EMPTY"
            )

    def fetch_daily_data(
        self,
        station_ids: Iterable[str],
        start_year: int,
        end_year: int | None = None,
    ) -> pl.DataFrame:
        """Fetch daily data using gap detection to minimize API calls."""
        all_dfs = []

        # Define requested range
        req_start = datetime.date(start_year, 1, 1)
        if end_year is None:
            req_end = datetime.date.today()
        else:
            req_end = datetime.date(end_year, 12, 31)
            today = datetime.date.today()
            if req_end > today:
                req_end = today

        for sid in station_ids:
            blocks = self._get_missing_blocks(sid, req_start, req_end)

            if not blocks:
                # We need the count of cached days for the log
                cached_dates = self.cache.get_existing_dates(
                    sid, req_start.isoformat(), req_end.isoformat()
                )
                logger.info(
                    f"[{sid}] 100% cache hit ({len(cached_dates)} days)"
                )
            else:
                existing_count = len(
                    self.cache.get_existing_dates(
                        sid, req_start.isoformat(), req_end.isoformat()
                    )
                )
                total_req = (req_end - req_start).days + 1
                logger.info(
                    f"[{sid}] Cache match: {existing_count}/{total_req} "
                    f"days found. Fetching {len(blocks)} gaps."
                )
                for b_start, b_end in blocks:
                    self._fetch_and_cache_block(sid, b_start, b_end)

            # Final retrieval from cache
            all_dfs.append(
                self.cache.get_cached_daily(sid, start_year, req_end.year)
            )

        if not all_dfs:
            return pl.DataFrame(schema=DAILY_SCHEMA)
        return pl.concat(all_dfs)
