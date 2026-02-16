"""Migration script for climate data cache optimization."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OLD_DB = "climate_cache.sq3"
NEW_DB = "climate_cache_new.sq3"


def date_to_int(date_str: str | None) -> int | None:
    """Convert YYYY-MM-DD or similar to YYYYMMDD integer."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        # Strip time or extra info (e.g., "1984-07-01 00:00:00" -> "1984-07-01")
        clean = date_str.split(" ")[0].split("T")[0]
        # Remove common separators
        clean = clean.replace("-", "").replace("/", "").replace(".", "")
        if len(clean) < 8:
            return None
        return int(clean[:8])
    except ValueError, TypeError, IndexError:
        return None


def scale_val(val: float | None) -> int | None:
    """Scale float by 10 and convert to integer."""
    if val is None:
        return None
    return int(round(val * 10))


def migrate(source_path: str, target_path: str) -> None:
    """Migrate data from old schema to new optimized schema."""
    if not Path(source_path).exists():
        logger.error(f"Source database {source_path} does not exist.")
        return

    # Ensure we don't overwrite if not intended (though NEW_DB is fine for this task)
    if Path(target_path).exists():
        logger.warning(
            f"Target database {target_path} already exists. It will be overwritten."
        )
        Path(target_path).unlink()

    with (
        sqlite3.connect(source_path) as conn_old,
        sqlite3.connect(target_path) as conn_new,
    ):
        # 1. Initialize New Schema
        logger.info("Initializing new schema...")
        conn_new.execute("""
            CREATE TABLE stations (
                key INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT UNIQUE,
                name TEXT,
                latitude REAL,
                longitude REAL
            )
        """)

        conn_new.execute("""
            CREATE TABLE daily_data (
                station_key INTEGER,
                date INTEGER,
                temp_mean INTEGER,
                temp_min INTEGER,
                temp_max INTEGER,
                precip INTEGER,
                PRIMARY KEY (station_key, date)
            ) WITHOUT ROWID
        """)

        conn_new.execute("""
            CREATE TABLE station_requests (
                station_key INTEGER,
                start_date INTEGER,
                end_date INTEGER,
                status TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Migrate Stations
        logger.info("Migrating stations...")
        cursor_old = conn_old.cursor()
        cursor_old.execute("SELECT id, name, latitude, longitude FROM stations")
        stations = cursor_old.fetchall()

        for sid, name, lat, lon in stations:
            conn_new.execute(
                "INSERT INTO stations (station_id, name, latitude, longitude) VALUES (?, ?, ?, ?)",
                (sid, name, lat, lon),
            )

        # Build lookup map: station_id (TEXT) -> key (INT)
        cursor_new = conn_new.cursor()
        cursor_new.execute("SELECT station_id, key FROM stations")
        id_to_key = dict(cursor_new.fetchall())

        # 3. Migrate Daily Data
        logger.info("Migrating daily observations...")
        cursor_old.execute(
            "SELECT station_id, date, temp_mean, temp_min, temp_max, precip FROM daily_data"
        )

        # Batch processing for efficiency if the DB is large
        batch_size = 10000
        while True:
            rows = cursor_old.fetchmany(batch_size)
            if not rows:
                break

            new_rows = []
            for sid, d_str, t_mean, t_min, t_max, pr in rows:
                s_key = id_to_key.get(sid)
                d_int = date_to_int(d_str)
                if s_key is not None and d_int is not None:
                    new_rows.append(
                        (
                            s_key,
                            d_int,
                            scale_val(t_mean),
                            scale_val(t_min),
                            scale_val(t_max),
                            scale_val(pr),
                        )
                    )

            conn_new.executemany(
                "INSERT OR IGNORE INTO daily_data (station_key, date, temp_mean, temp_min, temp_max, precip) VALUES (?, ?, ?, ?, ?, ?)",
                new_rows,
            )

        # 4. Migrate Station Requests
        logger.info("Migrating request logs...")
        cursor_old.execute(
            "SELECT station_id, start_date, end_date, status, timestamp FROM station_requests"
        )
        requests = cursor_old.fetchall()

        new_reqs = []
        for sid, s_date, e_date, status, ts in requests:
            s_key = id_to_key.get(sid)
            sd_int = date_to_int(s_date)
            ed_int = date_to_int(e_date)
            if s_key is not None:
                new_reqs.append((s_key, sd_int, ed_int, status, ts))

        conn_new.executemany(
            "INSERT INTO station_requests (station_key, start_date, end_date, status, timestamp) VALUES (?, ?, ?, ?, ?)",
            new_reqs,
        )

        conn_new.commit()
        logger.info("Migration complete.")


if __name__ == "__main__":
    migrate(OLD_DB, NEW_DB)
