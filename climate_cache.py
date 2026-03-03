"""SQLite caching for climate data with optimized schema."""

from __future__ import annotations

import datetime
import logging
import sqlite3
from typing import Any
from typing import Final

import polars as pl

import os

logger = logging.getLogger(__name__)

CACHE_DB: Final[str] = "climate_cache.sq3"

DAILY_SCHEMA: Final[dict[str, Any]] = {
    "station_id": pl.String,
    "date": pl.String,
    "temp_mean": pl.Float64,
    "temp_min": pl.Float64,
    "temp_max": pl.Float64,
    "precip_total": pl.Float64,
}


def _date_to_int(date_str: str | None) -> int | None:
    """Convert date string to YYYYMMDD integer."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        clean = date_str.split("T")[0].replace("-", "")
        if len(clean) >= 8:
            return int(clean[:8])
        return None
    except ValueError, TypeError, IndexError:
        return None


def _int_to_date_str(date_int: int | None) -> str | None:
    """Convert YYYYMMDD integer to date string."""
    if date_int is None:
        return None
    s = str(date_int)
    if len(s) != 8:
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


class ClimateCache:
    """Manages SQLite caching for daily climate data with optimized storage."""

    def __init__(self, db_path: str = CACHE_DB) -> None:
        """Initialize the cache."""
        if not os.path.isabs(db_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, db_path)
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
                    PRIMARY KEY (station_key, date),
                    FOREIGN KEY(station_key) REFERENCES stations(key)
                ) WITHOUT ROWID
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS station_periods (
                    station_key INTEGER,
                    start_date INTEGER,
                    end_date INTEGER,
                    PRIMARY KEY (station_key, start_date),
                    FOREIGN KEY(station_key) REFERENCES stations(key)
                )
                """
            )

    def _get_station_key(
        self, conn: sqlite3.Connection, station_id: str
    ) -> int:
        """Get or create internal key for a station."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key FROM stations WHERE station_id = ?", (station_id,)
        )
        row = cursor.fetchone()
        if row:
            return int(row[0])

        cursor.execute(
            "INSERT INTO stations (station_id) VALUES (?)", (station_id,)
        )
        return int(cursor.lastrowid)

    def get_missing_blocks(
        self, station_ids: list[str], start_year: int, end_year: int
    ) -> list[tuple[str, str, str]]:
        """Identify missing blocks of data across multiple stations."""

        blocks = []

        # Start and end boundaries as ISO strings
        req_start_iso = f"{start_year}-01-01"
        current_year = datetime.datetime.now().year
        effective_end_year = min(end_year, current_year)
        req_end_iso = f"{effective_end_year}-12-31"

        for sid in station_ids:
            sid_blocks = self._get_missing_blocks_for_station(
                sid, req_start_iso, req_end_iso
            )
            for start, end in sid_blocks:
                blocks.append((sid, start, end))
        return blocks

    def _get_missing_blocks_for_station(
        self, station_id: str, req_start: str, req_end: str
    ) -> list[tuple[str, str]]:
        """Identify gaps in data for a single station using simple interval subtraction."""
        start_d = datetime.date.fromisoformat(req_start)
        end_d = datetime.date.fromisoformat(req_end)

        blocks = []
        curr = start_d

        existing_periods = self.get_station_periods(station_id)

        for s, e in existing_periods:
            p_start = datetime.date.fromisoformat(s)
            p_end = datetime.date.fromisoformat(e)

            if p_end < curr:
                continue
            if p_start > end_d:
                break

            if p_start > curr:
                blocks.append(
                    (
                        curr.isoformat(),
                        (p_start - datetime.timedelta(days=1)).isoformat(),
                    )
                )

            curr = max(curr, p_end + datetime.timedelta(days=1))

        if curr <= end_d:
            blocks.append((curr.isoformat(), end_d.isoformat()))

        return blocks

    def save_stations(self, df: pl.DataFrame) -> None:
        """Save new stations."""
        if df.is_empty():
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO stations (station_id, name, latitude, longitude)
                VALUES (?, ?, ?, ?)
                """,
                df.select(["id", "name", "latitude", "longitude"]).rows(),
            )

    def _save_daily(self, df: pl.DataFrame, conn: sqlite3.Connection) -> None:
        """Internal bulk save using existing connection."""
        if df.is_empty():
            return

        df = df.with_columns(
            [
                (pl.col("temp_mean") * 10).round(0).cast(pl.Int64),
                (pl.col("temp_min") * 10).round(0).cast(pl.Int64),
                (pl.col("temp_max") * 10).round(0).cast(pl.Int64),
                (pl.col("precip_total") * 10).round(0).cast(pl.Int64),
                pl.col("date")
                .map_elements(_date_to_int, return_dtype=pl.Int64)
                .alias("date_int"),
            ]
        )

        # We process by station to get keys efficiently
        for (sid,), station_df in df.group_by("station_id"):
            station_key = self._get_station_key(conn, str(sid))

            rows = station_df.select(
                [
                    pl.lit(station_key),
                    pl.col("date_int"),
                    pl.col("temp_mean"),
                    pl.col("temp_min"),
                    pl.col("temp_max"),
                    pl.col("precip_total"),
                ]
            ).rows()

            conn.executemany(
                "INSERT OR REPLACE INTO daily_data VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def _add_station_period(
        self,
        station_id: str,
        start: str,
        end: str,
        conn: sqlite3.Connection,
    ) -> None:
        """Add a period and consolidate overlapping/adjacent ranges using Polars."""
        station_key = self._get_station_key(conn, station_id)

        # Fetch current rows
        rows = conn.execute(
            "SELECT start_date, end_date FROM station_periods WHERE station_key = ?",
            (station_key,),
        ).fetchall()

        # Add new range to list of dicts for Polars
        periods = [{"start": start, "end": end}]
        for r_start, r_end in rows:
            s_iso = _int_to_date_str(r_start)
            e_iso = _int_to_date_str(r_end)
            if s_iso and e_iso:
                periods.append({"start": s_iso, "end": e_iso})

        df = (
            pl.DataFrame(periods)
            .with_columns(
                [pl.col("start").str.to_date(), pl.col("end").str.to_date()]
            )
            .sort("start")
        )

        # Consolidation logic: Gaps are > 1 day
        merged = (
            df.with_columns(
                is_gap=(
                    pl.col("start")
                    > (
                        pl.col("end").shift().cum_max()
                        + datetime.timedelta(days=1)
                    )
                ).fill_null(True)
            )
            .with_columns(group=pl.col("is_gap").cum_sum())
            .group_by("group")
            .agg([pl.col("start").min(), pl.col("end").max()])
            .sort("start")
        )

        # Replace in DB
        conn.execute(
            "DELETE FROM station_periods WHERE station_key = ?", (station_key,)
        )
        for row in merged.iter_rows(named=True):
            s_int = _date_to_int(row["start"].isoformat())
            e_int = _date_to_int(row["end"].isoformat())
            conn.execute(
                "INSERT INTO station_periods (station_key, start_date, end_date) VALUES (?, ?, ?)",
                (station_key, s_int, e_int),
            )

    def save_daily_request(
        self, station_id: str, start: str, end: str, df: pl.DataFrame
    ) -> None:
        """Atomic save of both daily data and the request metadata."""
        with sqlite3.connect(self.db_path) as conn:
            if not df.is_empty():
                self._save_daily(df, conn)
            self._add_station_period(station_id, start, end, conn)

    def get_station_periods(self, station_id: str) -> list[tuple[str, str]]:
        """Retrieve request history for a station."""
        logger.debug(
            f"Fetching periods for station {station_id} from {self.db_path}"
        )

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT sp.start_date, sp.end_date
                FROM station_periods sp
                JOIN stations s ON sp.station_key = s.key
                WHERE s.station_id = ?
                ORDER BY sp.start_date
                """
                rows = conn.execute(query, (station_id,)).fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error in get_station_periods: {e}")
            return []

        results = []
        for r_start, r_end in rows:
            start_iso = _int_to_date_str(r_start)
            end_iso = _int_to_date_str(r_end)
            if start_iso and end_iso:
                results.append((start_iso, end_iso))

        if not results:
            logger.debug(f"No periods found for station {station_id}")
        else:
            logger.debug(
                f"Found {len(results)} periods for station {station_id}"
            )

        return results

    def get_daily_data(
        self, station_ids: list[str], start_year: int, end_year: int
    ) -> pl.DataFrame:
        """Retrieve daily data as a Polars DataFrame."""
        s_int = start_year * 10000 + 101
        e_int = end_year * 10000 + 1231

        query = """
        SELECT s.station_id, d.date, d.temp_mean / 10.0, d.temp_min / 10.0, d.temp_max / 10.0, d.precip / 10.0
        FROM daily_data d
        JOIN stations s ON d.station_key = s.key
        WHERE s.station_id IN ({}) AND d.date >= ? AND d.date <= ?
        """.format(",".join(["?"] * len(station_ids)))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (*station_ids, s_int, e_int))
            rows = cursor.fetchall()

        if not rows:
            return pl.DataFrame(schema=DAILY_SCHEMA)

        df = pl.DataFrame(
            rows,
            schema={
                "station_id": pl.String,
                "date_int": pl.Int64,
                "temp_mean": pl.Float64,
                "temp_min": pl.Float64,
                "temp_max": pl.Float64,
                "precip_total": pl.Float64,
            },
            orient="row",
        )

        # Convert date_int back to ISO
        df = df.with_columns(
            [
                pl.col("date_int")
                .map_elements(_int_to_date_str, return_dtype=pl.String)
                .alias("date"),
            ]
        )

        df = df.with_columns(
            [
                pl.col("date").str.to_date().dt.year().alias("year"),
                pl.col("date").str.to_date().dt.month().alias("month"),
                pl.col("date").str.to_date().dt.day().alias("day"),
            ]
        )

        return df

    def get_cache_summary(self) -> pl.DataFrame:
        """Summary of cached data for --cache-report."""
        query = """
        SELECT 
            s.station_id, 
            s.name, 
            sp.start_date, 
            sp.end_date,
            (
                SELECT COUNT(*) 
                FROM daily_data d 
                WHERE d.station_key = s.key 
                  AND d.date >= sp.start_date 
                  AND d.date <= sp.end_date
            ) as days_within_period
        FROM station_periods sp
        JOIN stations s ON sp.station_key = s.key
        ORDER BY s.station_id, sp.start_date
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(
            rows,
            schema=[
                "station_id",
                "name",
                "start_date",
                "end_date",
                "days_cached",
            ],
            orient="row",
        )

        df = df.with_columns(
            [
                pl.col("start_date").map_elements(
                    lambda x: _int_to_date_str(x),
                    return_dtype=pl.String,
                ),
                pl.col("end_date").map_elements(
                    lambda x: _int_to_date_str(x),
                    return_dtype=pl.String,
                ),
            ]
        )
        return df
