"""SQLite caching for climate data."""

from __future__ import annotations

import sqlite3
from typing import Any
from typing import Final

import polars as pl

CACHE_DB: Final[str] = "climate_cache.sq3"

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS station_requests (
                    station_id TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    (
                        row["id"],
                        row["name"],
                        row["latitude"],
                        row["longitude"],
                    ),
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

    def get_existing_dates(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> set[str]:
        """Get set of dates already in the database for a station/range."""
        query = """
            SELECT date FROM daily_data
            WHERE station_id = ? AND date BETWEEN ? AND ?
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (station_id, start_date, end_date))
            return {r[0] for r in cursor.fetchall()}

    def log_station_request(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        status: str,
    ) -> None:
        """Log a successful fetch request to skip future redundant checks."""
        query = """
            INSERT INTO station_requests
            (station_id, start_date, end_date, status)
            VALUES (?, ?, ?, ?)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, (station_id, start_date, end_date, status))

    def get_station_requests(
        self, station_id: str
    ) -> list[tuple[str, str, str]]:
        """Get past request ranges for a station."""
        query = """
            SELECT start_date, end_date, status
            FROM station_requests
            WHERE station_id = ?
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (station_id,))
            return cursor.fetchall()

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
