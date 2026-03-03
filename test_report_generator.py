import datetime

import polars as pl
import pytest

from report_generator import _calculate_period_stats
from report_generator import aggregate_data


@pytest.fixture
def sample_daily_df():
    """Create sample daily data for a station."""
    # Generating 2 years of constant data (2020 and 2021)
    dates = []
    # 2020 (Leap year: 366 days)
    start_2020 = datetime.date(2020, 1, 1)
    for i in range(366):
        dates.append((start_2020 + datetime.timedelta(days=i)).isoformat())

    # 2021 (365 days)
    start_2021 = datetime.date(2021, 1, 1)
    for i in range(365):
        dates.append((start_2021 + datetime.timedelta(days=i)).isoformat())

    df = pl.DataFrame(
        {
            "station_id": ["S1"] * (366 + 365),
            "date": dates,
            "temp_mean": [5.0] * (366 + 365),
            "temp_min": [0.0] * (366 + 365),
            "temp_max": [10.0] * (366 + 365),
            "precip_total": [1.0] * (366 + 365),
        }
    )

    # Add year/month/day
    df = df.with_columns(
        [
            pl.col("date").str.to_date().dt.year().alias("year"),
            pl.col("date").str.to_date().dt.month().alias("month"),
            pl.col("date").str.to_date().dt.day().alias("day"),
        ]
    )
    return df


@pytest.fixture
def sample_stations_df():
    return pl.DataFrame(
        {
            "id": ["S1"],
            "name": ["Test Station"],
            "latitude": [45.5],
            "longitude": [-73.5],
            "requested_location": ["Montreal, Canada"],
        }
    )


def test_aggregate_data_monthly(sample_daily_df, sample_stations_df):
    """Verify daily -> monthly aggregation."""
    agg_df = aggregate_data(
        sample_daily_df,
        sample_stations_df,
        max_temp=False,
        min_temp=False,
        period="monthly",
    )

    # Expect 12 months for 2 years = 24 rows total per station
    # (The aggregation groups by station_id, requested_location, period_name, year, period_idx)
    assert len(agg_df) == 24

    # Check Jan 2020 (temp_mean = 5.0 everywhere)
    jan_2020 = agg_df.filter(
        (pl.col("year") == 2020) & (pl.col("period_idx") == 1)
    )
    assert jan_2020["temp_mean"][0] == 5.0
    # Precip: 1mm/day * 31 days = 31mm
    assert jan_2020["precip_total"][0] == 31.0


def test_aggregate_data_seasonal(sample_daily_df, sample_stations_df):
    """Verify daily -> seasonal aggregation."""
    agg_df = aggregate_data(
        sample_daily_df,
        sample_stations_df,
        max_temp=False,
        min_temp=False,
        period="seasonally",
    )

    # Expect 4 seasons for 2 years = 8 rows
    assert len(agg_df) == 8

    # Winter (DJF) 2020: 31 days Jan + 29 days Feb + (prev Dec omitted)
    winter_2020 = agg_df.filter(
        (pl.col("year") == 2020) & (pl.col("period_idx") == 1)
    )
    # Correct period count for aggregation depends on implementation.
    # Current implementation groups (month 12 + month 1 + month 2)
    # 2020 Winter index 1 is Jan/Feb/Dec 2020. Since Dec 2020 has 1mm, total should be ~60mm (omitting 2019 Dec)
    assert winter_2020["precip_total"][0] >= 60.0  # 31 + 29 + 31


def test_calculate_period_stats_metrics(sample_daily_df, sample_stations_df):
    """Verify stats like min/max/avg for temp."""
    agg_df = aggregate_data(
        sample_daily_df, sample_stations_df, period="monthly"
    )

    # Jan stats for loc1
    loc1 = "Montreal, Canada"
    stats_df = _calculate_period_stats(agg_df, 1, "temperature", location=loc1)

    assert stats_df is not None
    assert "avg" in stats_df.columns
    assert "anomaly" in stats_df.columns
    assert "trend" in stats_df.columns

    # Years: 2020, 2021
    assert len(stats_df) == 2
    # temp_mean was 5.0 everywhere, avg should be 5.0
    assert abs(stats_df["avg"][0] - 5.0) < 0.001
    # Multi-year average (lt_mean) is 5.0, trend is flat at 5.0, so anomaly is 0.0
    assert abs(stats_df["anomaly"][0]) < 0.001


def test_calculate_period_stats_with_no_data():
    """Verify it returns None if no rows for period_idx."""
    empty_df = pl.DataFrame(
        schema={
            "period_idx": pl.Int16,
            "requested_location": pl.String,
            "year": pl.Int16,
            "temp_mean": pl.Float64,
        }
    )
    res = _calculate_period_stats(empty_df, 1, "temperature")
    assert res is None
