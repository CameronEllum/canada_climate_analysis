"""SQLite caching for climate data with optimized schema."""

from __future__ import annotations

import sqlite3
from typing import Any
from typing import Final

import polars as pl

CACHE_DB: Final[str] = "climate_cache_new.sq3"

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
    """Manages SQLite caching for daily climate data with optimized storage."""

    def __init__(self, db_path: str = CACHE_DB) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite schema using optimized types."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stations (
                    key INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT UNIQUE,
                    name TEXT,
                    latitude REAL,
                    longitude REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_data (
                    station_key INTEGER,
                    date INTEGER, -- YYYYMMDD
                    temp_mean INTEGER, -- Scaled by 10
                    temp_min INTEGER, -- Scaled by 10
                    temp_max INTEGER, -- Scaled by 10
                    precip INTEGER, -- Scaled by 10
                    PRIMARY KEY (station_key, date)
                ) WITHOUT ROWID
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS station_requests (
                    station_key INTEGER,
                    start_date INTEGER,
                    end_date INTEGER,
                    status TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _get_station_key(
        self, conn: sqlite3.Connection, station_id: str
    ) -> int | None:
        """Lookup internal integer key for a station ID."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key FROM stations WHERE station_id = ?", (station_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _date_to_int(self, date_str: str | None) -> int | None:
        """Convert ISO date string to YYYYMMDD integer."""
        if not date_str or not isinstance(date_str, str):
            return None
        try:
            # Robust parsing (handles "YYYY-MM-DD", "YYYY-MM-DD HH:MM:SS", etc.)
            clean = date_str.split(" ")[0].split("T")[0]
            clean = clean.replace("-", "").replace("/", "").replace(".", "")
            if len(clean) < 8:
                return None
            return int(clean[:8])
        except ValueError, TypeError, IndexError:
            return None

    def _int_to_date_components(
        self, date_int: int | None
    ) -> tuple[str, int, int, int] | tuple[None, None, None, None]:
        """Convert YYYYMMDD integer to (ISO string, year, month, day)."""
        if date_int is None:
            return None, None, None, None
        s = str(date_int)
        if len(s) != 8:
            return None, None, None, None
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:])
        return f"{y:04d}-{m:02d}-{d:02d}", y, m, d

    def save_stations(self, df: pl.DataFrame) -> None:
        """Save station metadata to cache."""
        with sqlite3.connect(self.db_path) as conn:
            for row in df.iter_rows(named=True):
                conn.execute(
                    """
                    INSERT INTO stations (station_id, name, latitude, longitude)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(station_id) DO UPDATE SET
                        name=excluded.name,
                        latitude=excluded.latitude,
                        longitude=excluded.longitude
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
        with sqlite3.connect(self.db_path) as conn:
            s_key = self._get_station_key(conn, station_id)
            if s_key is None:
                return pl.DataFrame(schema=DAILY_SCHEMA)

            query = """
                SELECT date, temp_mean, temp_min, temp_max, precip
                FROM daily_data
                WHERE station_key = ? AND date BETWEEN ? AND ?
            """
            # start_year to YYYYMMDD
            s_date = start_year * 10000 + 101
            e_date = end_year * 10000 + 1231

            cursor = conn.cursor()
            cursor.execute(query, (s_key, s_date, e_date))
            rows = cursor.fetchall()
            if not rows:
                return pl.DataFrame(schema=DAILY_SCHEMA)

            data = []
            for r in rows:
                d_int = r[0]
                d_str, y, m, d = self._int_to_date_components(d_int)
                data.append(
                    {
                        "station_id": station_id,
                        "date": d_str,
                        "year": y,
                        "month": m,
                        "day": d,
                        "temp_mean": r[1] / 10.0 if r[1] is not None else None,
                        "temp_min": r[2] / 10.0 if r[2] is not None else None,
                        "temp_max": r[3] / 10.0 if r[3] is not None else None,
                        "precip": r[4] / 10.0 if r[4] is not None else None,
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
        with sqlite3.connect(self.db_path) as conn:
            s_key = self._get_station_key(conn, station_id)
            if s_key is None:
                return set()

            s_date_int = self._date_to_int(start_date)
            e_date_int = self._date_to_int(end_date)

            query = """
                SELECT date FROM daily_data
                WHERE station_key = ? AND date BETWEEN ? AND ?
            """
            cursor = conn.cursor()
            cursor.execute(query, (s_key, s_date_int, e_date_int))
            # Convert back to ISO string if needed by MSCClient
            return {
                self._int_to_date_components(r[0])[0] for r in cursor.fetchall()
            }

    def log_station_request(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        status: str,
    ) -> None:
        """Log a successful fetch request."""
        with sqlite3.connect(self.db_path) as conn:
            s_key = self._get_station_key(conn, station_id)
            if s_key is None:
                return

            query = """
                INSERT INTO station_requests
                (station_key, start_date, end_date, status)
                VALUES (?, ?, ?, ?)
            """
            conn.execute(
                query,
                (
                    s_key,
                    self._date_to_int(start_date),
                    self._date_to_int(end_date),
                    status,
                ),
            )

    def get_station_requests(
        self, station_id: str
    ) -> list[tuple[str, str, str]]:
        """Get past request ranges for a station."""
        with sqlite3.connect(self.db_path) as conn:
            s_key = self._get_station_key(conn, station_id)
            if s_key is None:
                return []

            query = """
                SELECT start_date, end_date, status
                FROM station_requests
                WHERE station_key = ?
            """
            cursor = conn.cursor()
            cursor.execute(query, (s_key,))
            results = []
            for r in cursor.fetchall():
                # Convert back to ISO strings for MSCClient comparison logic
                s_str = self._int_to_date_components(r[0])[0]
                e_str = self._int_to_date_components(r[1])[0]
                results.append((s_str, e_str, r[2]))
            return results

    def save_daily(self, df: pl.DataFrame) -> None:
        """Save daily observations to cache using scaling."""
        if df.is_empty():
            return

        with sqlite3.connect(self.db_path) as conn:
            # Group by station to minimize lookups.
            # Note: Polars group_by iteration returns (key_tuple, group_df)
            for key_tuple, group in df.group_by("station_id"):
                sid = (
                    key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple
                )
                s_key = self._get_station_key(conn, sid)  # type: ignore
                if s_key is None:
                    continue

                rows_to_insert = []
                for row in group.iter_rows(named=True):
                    d_int = self._date_to_int(row["date"])
                    t_mean = (
                        int(round(row["temp_mean"] * 10))
                        if row["temp_mean"] is not None
                        else None
                    )
                    t_min = (
                        int(round(row["temp_min"] * 10))
                        if row["temp_min"] is not None
                        else None
                    )
                    t_max = (
                        int(round(row["temp_max"] * 10))
                        if row["temp_max"] is not None
                        else None
                    )
                    precip = (
                        int(round(row["precip"] * 10))
                        if row["precip"] is not None
                        else None
                    )

                    rows_to_insert.append(
                        (s_key, d_int, t_mean, t_min, t_max, precip)
                    )

                conn.executemany(
                    """
                    INSERT INTO daily_data
                    (station_key, date, temp_mean, temp_min, temp_max, precip)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(station_key, date) DO UPDATE SET
                        temp_mean=excluded.temp_mean,
                        temp_min=excluded.temp_min,
                        temp_max=excluded.temp_max,
                        precip=excluded.precip
                    """,
                    rows_to_insert,
                )
