import sqlite3

import polars as pl
import pytest

from climate_cache import DAILY_SCHEMA
from climate_cache import ClimateCache


@pytest.fixture
def temp_cache(tmp_path):
    db_path = tmp_path / "test_cache.sq3"
    cache = ClimateCache(str(db_path))
    return cache


def test_init_db(temp_cache):
    """Verify that tables are created."""
    with sqlite3.connect(temp_cache.db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "stations" in table_names
        assert "daily_data" in table_names
        assert "station_periods" in table_names


def test_missing_blocks_empty_cache(temp_cache):
    """Test gap detection with completely empty cache."""
    sid = "TEST-01"
    start_year = 2020
    end_year = 2020

    blocks = temp_cache.get_missing_blocks([sid], start_year, end_year)

    # Expect one block for the full year
    assert len(blocks) == 1
    assert blocks[0] == (sid, "2020-01-01", "2020-12-31")


def test_missing_blocks_partial_hit(temp_cache):
    """Test gap detection when cache contains SOME data."""
    sid = "TEST-01"

    # Save some data for Jan 2020
    data = [
        {
            "station_id": sid,
            "date": "2020-01-01",
            "temp_mean": 5.0,
            "temp_min": 0.0,
            "temp_max": 10.0,
            "precip_total": 0.0,
        }
    ]
    df = pl.DataFrame(data, schema=DAILY_SCHEMA)
    temp_cache.save_daily_request(sid, "2020-01-01", "2020-01-01", df)

    # Verify it identified the gap (rest of 2020)
    # The current logic GROUPS missing dates into blocks.
    # 2020-01-02 to 2020-12-31 should be the block.
    blocks = temp_cache.get_missing_blocks([sid], 2020, 2020)

    assert len(blocks) == 1
    assert blocks[0] == (sid, "2020-01-02", "2020-12-31")


def test_missing_blocks_with_logged_requests(temp_cache):
    """Test that it doesn't re-request ranges marked as SUCCESS (or EMPTY)."""
    sid = "TEST-01"

    # Log a request from 2020-01-01 to 2020-01-31 as successful
    temp_cache.save_daily_request(
        sid, "2020-01-01", "2020-01-31", pl.DataFrame(schema=DAILY_SCHEMA)
    )

    # Requesting 2020-01-01 to 2020-02-15 should only return the new gap in Feb
    blocks = temp_cache.get_missing_blocks(
        [sid], 2020, 2020
    )  # This will check full year

    # Actually get_missing_blocks for 2020 start_year/end_year defaults to full ISO range
    # Let me call _get_missing_blocks_for_station directly for precision
    p_blocks = temp_cache._get_missing_blocks_for_station(
        sid, "2020-01-01", "2020-02-15"
    )

    assert len(p_blocks) == 1
    assert p_blocks[0] == ("2020-02-01", "2020-02-15")


def test_save_and_retrieve_daily(temp_cache):
    """Test roundtrip of daily data."""
    sid = "TEST-01"
    data = [
        {
            "station_id": sid,
            "date": "2020-01-01",
            "temp_mean": 1.555,
            "temp_min": -5.111,
            "temp_max": 10.222,
            "precip_total": 15.0,
        },
        {
            "station_id": sid,
            "date": "2020-01-02",
            "temp_mean": 2.0,
            "temp_min": -4.0,
            "temp_max": 11.0,
            "precip_total": 0.0,
        },
    ]
    df = pl.DataFrame(data, schema=DAILY_SCHEMA)
    temp_cache.save_daily_request(sid, "2020-01-01", "2020-01-02", df)

    # Retrieve
    retrieved = temp_cache.get_daily_data([sid], 2020, 2020)

    assert len(retrieved) == 2
    # Verify scaling (precision check)
    row1 = retrieved.filter(pl.col("date") == "2020-01-01")
    assert round(row1["temp_mean"][0], 1) == 1.6
    assert round(row1["temp_min"][0], 1) == -5.1
    assert round(row1["temp_max"][0], 1) == 10.2
    assert row1["precip_total"][0] == 15.0


def test_cache_summary(temp_cache):
    """Test generating a summary."""
    temp_cache.save_daily_request(
        "S1",
        "2010-01-01",
        "2010-01-02",
        pl.DataFrame(
            [
                {
                    "station_id": "S1",
                    "date": "2010-01-01",
                    "temp_mean": 1.0,
                    "temp_min": 0.0,
                    "temp_max": 2.0,
                    "precip_total": 0.0,
                },
                {
                    "station_id": "S1",
                    "date": "2010-01-02",
                    "temp_mean": 2.0,
                    "temp_min": 1.0,
                    "temp_max": 3.0,
                    "precip_total": 0.0,
                },
            ],
            schema=DAILY_SCHEMA,
        ),
    )

    summary = temp_cache.get_cache_summary()
    assert len(summary) == 1
    assert summary["days_cached"][0] == 2
    assert summary["start_date"][0] == "2010-01-01"
    assert summary["end_date"][0] == "2010-01-02"


def test_period_consolidation(temp_cache):
    """Verify that overlapping/adjacent periods are merged."""
    sid = "TEST-01"

    # Add two overlapping periods
    temp_cache.save_daily_request(
        sid, "2020-01-01", "2020-01-15", pl.DataFrame(schema=DAILY_SCHEMA)
    )
    temp_cache.save_daily_request(
        sid, "2020-01-10", "2020-01-20", pl.DataFrame(schema=DAILY_SCHEMA)
    )

    periods = temp_cache.get_station_periods(sid)
    assert len(periods) == 1
    assert periods[0] == ("2020-01-01", "2020-01-20")

    # Add an adjacent period (Jan 21)
    temp_cache.save_daily_request(
        sid, "2020-01-21", "2020-01-31", pl.DataFrame(schema=DAILY_SCHEMA)
    )
    periods = temp_cache.get_station_periods(sid)
    assert len(periods) == 1
    assert periods[0] == ("2020-01-01", "2020-01-31")

    # Add a disjoint period (Feb 15)
    temp_cache.save_daily_request(
        sid, "2020-02-15", "2020-02-20", pl.DataFrame(schema=DAILY_SCHEMA)
    )
    periods = temp_cache.get_station_periods(sid)
    assert len(periods) == 2
    assert periods[0] == ("2020-01-01", "2020-01-31")
    assert periods[1] == ("2020-02-15", "2020-02-20")
